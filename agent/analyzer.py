"""
Cross-Container Analyzer.
Uses baseline vs contention comparison to identify noisy neighbors.
"""
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from agent.collector import MetricsCollector
from agent.rag import ConfigKnowledgeBase


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


class NoisyNeighborDetector:
    """
    Detects noisy neighbors by comparing baseline to contention periods.
    
    Key insight: the noisy neighbor's CPU delta (from its own baseline) is 
    the highest. Victims show latency delta without CPU delta.
    Absolute metrics are misleading because cpu_shares create different 
    baseline CPU levels per container.
    """

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def detect(self, baseline_window: int = 10, contention_window: int = 20) -> List[Dict]:
        """
        Compare baseline period to contention period for each container.
        baseline_window: last N samples of the baseline period
        contention_window: last N samples of the contention period
        """
        deltas = {}

        for name in self.collector.CONTAINER_NAMES:
            metrics_df = self.collector.metrics_store.get(name, pd.DataFrame())
            latency_df = self.collector.latency_store.get(name, pd.DataFrame())

            if metrics_df.empty or latency_df.empty:
                continue
            if "latency_ms" not in latency_df.columns:
                continue

            total_samples = len(metrics_df)
            if total_samples < baseline_window + contention_window:
                # Not enough data, use what we have
                split = total_samples // 3  # First third = baseline
                if split < 3:
                    continue
                baseline_cpu = metrics_df["cpu_percent"].head(split)
                contention_cpu = metrics_df["cpu_percent"].tail(split * 2)
            else:
                # Baseline = first baseline_window samples
                baseline_cpu = metrics_df["cpu_percent"].head(baseline_window)
                # Contention = last contention_window samples
                contention_cpu = metrics_df["cpu_percent"].tail(contention_window)

            total_lat_samples = len(latency_df)
            if total_lat_samples < baseline_window + contention_window:
                split = total_lat_samples // 3
                if split < 3:
                    continue
                baseline_lat = latency_df["latency_ms"].head(split).dropna()
                contention_lat = latency_df["latency_ms"].tail(split * 2).dropna()
            else:
                baseline_lat = latency_df["latency_ms"].head(baseline_window).dropna()
                contention_lat = latency_df["latency_ms"].tail(contention_window).dropna()

            if baseline_cpu.empty or contention_cpu.empty:
                continue
            if baseline_lat.empty or contention_lat.empty:
                continue

            b_cpu = baseline_cpu.mean()
            c_cpu = contention_cpu.mean()
            b_lat = baseline_lat.median()
            c_lat = contention_lat.median()

            cpu_delta = c_cpu - b_cpu
            cpu_delta_pct = (cpu_delta / max(b_cpu, 0.1)) * 100
            lat_delta = c_lat - b_lat
            lat_delta_pct = (lat_delta / max(b_lat, 1.0)) * 100

            deltas[name] = {
                "baseline_cpu": round(b_cpu, 2),
                "contention_cpu": round(c_cpu, 2),
                "cpu_delta": round(cpu_delta, 2),
                "cpu_delta_pct": round(cpu_delta_pct, 1),
                "baseline_latency": round(b_lat, 2),
                "contention_latency": round(c_lat, 2),
                "lat_delta": round(lat_delta, 2),
                "lat_delta_pct": round(lat_delta_pct, 1),
            }

        if not deltas:
            return []

        # --- Attribution Logic ---
        # The noisy neighbor is the container with the largest CPU delta
        # from its own baseline. It's eating more CPU than before.
        # Victims have latency delta but NOT cpu delta.

        # Rank by CPU delta (descending)
        ranked_by_cpu_delta = sorted(
            deltas.items(), key=lambda x: x[1]["cpu_delta"], reverse=True
        )

        # The top CPU delta container is our primary suspect
        suspect_name, suspect_deltas = ranked_by_cpu_delta[0]

        # Only proceed if the suspect's CPU actually increased meaningfully
        if suspect_deltas["cpu_delta"] < 3:
            # No container had a meaningful CPU increase - no noisy neighbor
            return []

        # Find victims: containers with latency increase but small CPU delta
        detections = []
        for name, d in deltas.items():
            if name == suspect_name:
                continue

            # Victim criteria: latency went up, CPU did NOT go up much
            is_victim = (
                d["lat_delta_pct"] > 10  # Latency increased >10% from baseline
                and d["cpu_delta"] < suspect_deltas["cpu_delta"] * 0.5  # CPU didn't spike like suspect
            )

            if is_victim:
                confidence = self._calc_confidence(
                    suspect_deltas, d, deltas
                )
                detections.append({
                    "victim": {
                        "name": name,
                        "baseline_latency": d["baseline_latency"],
                        "contention_latency": d["contention_latency"],
                        "lat_delta_pct": d["lat_delta_pct"],
                        "own_cpu_delta": d["cpu_delta"],
                        "own_cpu_mean": d["contention_cpu"],
                    },
                    "suspect": {
                        "name": suspect_name,
                        "baseline_cpu": suspect_deltas["baseline_cpu"],
                        "contention_cpu": suspect_deltas["contention_cpu"],
                        "cpu_delta": suspect_deltas["cpu_delta"],
                        "cpu_delta_pct": suspect_deltas["cpu_delta_pct"],
                        "cpu_mean": suspect_deltas["contention_cpu"],
                        "cpu_max": suspect_deltas["contention_cpu"],  # approx
                        "mem_mean": 0,  # filled by caller if needed
                    },
                    "temporal_correlation": self._check_temporal_correlation(
                        name, suspect_name
                    ),
                    "confidence": confidence,
                    "deltas_summary": deltas,
                    "timestamp": datetime.now().isoformat(),
                })

        # If no victims found but suspect had big CPU delta, 
        # report anyway with the container that had the most latency increase
        if not detections and suspect_deltas["cpu_delta"] >= 3:
            # Pick the container with highest latency delta as victim
            non_suspect = {k: v for k, v in deltas.items() if k != suspect_name}
            if non_suspect:
                worst_victim_name = max(
                    non_suspect.items(), key=lambda x: x[1]["lat_delta_pct"]
                )
                vname, vd = worst_victim_name
                detections.append({
                    "victim": {
                        "name": vname,
                        "baseline_latency": vd["baseline_latency"],
                        "contention_latency": vd["contention_latency"],
                        "lat_delta_pct": vd["lat_delta_pct"],
                        "own_cpu_delta": vd["cpu_delta"],
                        "own_cpu_mean": vd["contention_cpu"],
                    },
                    "suspect": {
                        "name": suspect_name,
                        "baseline_cpu": suspect_deltas["baseline_cpu"],
                        "contention_cpu": suspect_deltas["contention_cpu"],
                        "cpu_delta": suspect_deltas["cpu_delta"],
                        "cpu_delta_pct": suspect_deltas["cpu_delta_pct"],
                        "cpu_mean": suspect_deltas["contention_cpu"],
                        "cpu_max": suspect_deltas["contention_cpu"],
                        "mem_mean": 0,
                    },
                    "temporal_correlation": self._check_temporal_correlation(
                        vname, suspect_name
                    ),
                    "confidence": self._calc_confidence(
                        suspect_deltas, vd, deltas
                    ),
                    "deltas_summary": deltas,
                    "timestamp": datetime.now().isoformat(),
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections

    def _check_temporal_correlation(self, victim_name: str, suspect_name: str) -> float:
        """Check temporal correlation between suspect CPU and victim latency."""
        suspect_df = self.collector.metrics_store.get(suspect_name, pd.DataFrame())
        victim_lat = self.collector.latency_store.get(victim_name, pd.DataFrame())

        if suspect_df.empty or victim_lat.empty:
            return 0.0

        min_len = min(len(suspect_df), len(victim_lat))
        if min_len < 5:
            return 0.0

        cpu = suspect_df["cpu_percent"].tail(min_len).reset_index(drop=True)
        lat = victim_lat["latency_ms"].tail(min_len).reset_index(drop=True)
        valid = pd.DataFrame({"cpu": cpu, "lat": lat}).dropna()

        if len(valid) < 5:
            return 0.0
        if valid["cpu"].std() < 0.01 or valid["lat"].std() < 0.01:
            return 0.3 if valid["cpu"].mean() > 20 else 0.0

        try:
            corr = valid["cpu"].corr(valid["lat"])
            return round(corr, 3) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _calc_confidence(
        self, suspect_d: dict, victim_d: dict, all_deltas: dict
    ) -> float:
        """Confidence based on how clearly the suspect stands out."""
        score = 0.0

        # How much did suspect's CPU increase vs others?
        all_cpu_deltas = [d["cpu_delta"] for d in all_deltas.values()]
        max_delta = max(all_cpu_deltas)
        second_max = sorted(all_cpu_deltas, reverse=True)[1] if len(all_cpu_deltas) > 1 else 0

        # Suspect stands out clearly from the pack
        if max_delta > 0 and second_max >= 0:
            separation = (max_delta - second_max) / max(max_delta, 1)
            score += separation * 40  # Up to 40 points

        # Absolute CPU increase of suspect
        if suspect_d["cpu_delta"] > 15:
            score += 25
        elif suspect_d["cpu_delta"] > 8:
            score += 15
        elif suspect_d["cpu_delta"] > 3:
            score += 8

        # Victim's latency increased meaningfully
        if victim_d["lat_delta_pct"] > 50:
            score += 20
        elif victim_d["lat_delta_pct"] > 20:
            score += 10

        # Victim's CPU did NOT increase (external cause)
        if abs(victim_d["cpu_delta"]) < 3:
            score += 15
        elif abs(victim_d["cpu_delta"]) < 5:
            score += 8

        return min(round(score, 1), 100)


class LLMAnalyzer:
    """Uses Ollama LLM + RAG to generate diagnosis and recommendations."""

    def __init__(self, knowledge_base: ConfigKnowledgeBase):
        self.kb = knowledge_base
        self.change_history: List[Dict] = []

    def diagnose(self, detection: Dict, all_summaries: Dict) -> Dict:
        victim = detection["victim"]
        suspect = detection["suspect"]

        rag_context = self.kb.query_for_diagnosis(victim["name"], suspect["name"])
        prompt = self._build_diagnosis_prompt(detection, all_summaries, rag_context)

        try:
            response = self._call_ollama(prompt)
        except Exception as e:
            response = (
                f"LLM unavailable ({e}). Metrics-based diagnosis: "
                f"{suspect['name']} (CPU delta: +{suspect['cpu_delta']:.1f}%) "
                f"is the likely noisy neighbor causing latency degradation in "
                f"{victim['name']} (latency: {victim['baseline_latency']:.0f}ms → "
                f"{victim['contention_latency']:.0f}ms, +{victim['lat_delta_pct']:.0f}%)."
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "victim": victim["name"],
            "suspect": suspect["name"],
            "confidence": detection["confidence"],
            "temporal_correlation": detection["temporal_correlation"],
            "diagnosis": response,
            "metrics_evidence": {
                "victim_baseline_latency": victim["baseline_latency"],
                "victim_contention_latency": victim["contention_latency"],
                "victim_lat_delta_pct": victim["lat_delta_pct"],
                "suspect_cpu_delta": suspect["cpu_delta"],
                "suspect_cpu_delta_pct": suspect["cpu_delta_pct"],
                "suspect_baseline_cpu": suspect["baseline_cpu"],
                "suspect_contention_cpu": suspect["contention_cpu"],
            },
        }

    def _build_diagnosis_prompt(
        self, detection: Dict, all_summaries: Dict, rag_context: str
    ) -> str:
        victim = detection["victim"]
        suspect = detection["suspect"]
        deltas = detection.get("deltas_summary", {})

        history_text = ""
        if self.change_history:
            recent = self.change_history[-5:]
            history_text = "\n=== RECENT CHANGE HISTORY ===\n"
            for h in recent:
                history_text += (
                    f"- {h['timestamp']}: {h['action']} on "
                    f"{h['container']} -> {h['outcome']}\n"
                )

        deltas_text = "\n=== BASELINE vs CONTENTION DELTAS ===\n"
        for name, d in deltas.items():
            deltas_text += (
                f"  {name}: CPU {d['baseline_cpu']:.1f}% → {d['contention_cpu']:.1f}% "
                f"(delta: {d['cpu_delta']:+.1f}%) | "
                f"Latency {d['baseline_latency']:.0f}ms → {d['contention_latency']:.0f}ms "
                f"(delta: {d['lat_delta_pct']:+.0f}%)\n"
            )

        return f"""You are an infrastructure operations agent. Analyze this noisy neighbor detection and recommend a specific config change.

=== DETECTION ===
Victim: {victim['name']}
  Latency: {victim['baseline_latency']:.0f}ms → {victim['contention_latency']:.0f}ms (+{victim['lat_delta_pct']:.0f}%)
  CPU delta: {victim['own_cpu_delta']:+.1f}% (small = external cause)

Suspect noisy neighbor: {suspect['name']}
  CPU: {suspect['baseline_cpu']:.1f}% → {suspect['contention_cpu']:.1f}% (delta: +{suspect['cpu_delta']:.1f}%)
  Correlation with victim latency: {detection['temporal_correlation']:.3f}
{deltas_text}
=== CONFIGURATION KNOWLEDGE (from RAG) ===
{rag_context}
{history_text}

Respond in this format:
ROOT CAUSE: [one sentence]
EVIDENCE: [2-3 bullet points]
RECOMMENDED CONFIG CHANGE: [specific change with values]
RISK: [potential downside]
ALTERNATIVE: [backup approach]"""

    def _call_ollama(self, prompt: str, timeout: int = 60) -> str:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "No response")

    def record_feedback(self, container: str, action: str, outcome: str):
        self.change_history.append({
            "timestamp": datetime.now().isoformat(),
            "container": container,
            "action": action,
            "outcome": outcome,
        })


class BaselineAnalyzer:
    """
    System A: Per-container threshold monitoring. No cross-correlation.
    Alerts on high metrics but blames whichever container shows symptoms.
    """

    def detect(self, summaries: Dict) -> List[Dict]:
        alerts = []
        for name, stats in summaries.items():
            if not stats:
                continue
            if stats.get("cpu_mean", 0) > 50:
                alerts.append({
                    "container": name,
                    "alert": "HIGH_CPU",
                    "value": stats["cpu_mean"],
                    "diagnosis": f"{name} has high CPU. Investigate {name}.",
                })
            if stats.get("mem_mean", 0) > 70:
                alerts.append({
                    "container": name,
                    "alert": "HIGH_MEMORY",
                    "value": stats["mem_mean"],
                    "diagnosis": f"{name} has high memory. Investigate {name}.",
                })
            latency = stats.get("latency_mean")
            if latency and latency > 50:
                alerts.append({
                    "container": name,
                    "alert": "HIGH_LATENCY",
                    "value": latency,
                    "diagnosis": f"{name} has high latency ({latency:.0f}ms). Investigate {name}.",
                })
        return alerts
