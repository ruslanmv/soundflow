# generator/free/pattern_engine.py
"""
SoundFlow Pattern Engine – v4.0
Musical intelligence: phrases, motif reuse, mutation, groove, harmony coupling.

Key upgrades:
- 64-step phrase generation (multi-bar)
- mutation system (controlled evolution)
- groove profiles (house/techno/jazz swing triplet)
- key-aware chord progressions with voicings
- melody/harmony coupling (chord-tone preference + boundary resolution)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

# =============================================================================
# MUSICAL CONSTANTS
# =============================================================================

SCALES: Dict[str, List[int]] = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "acid": [0, 3, 7, 10],  # simplified; acid behavior is in melody logic
    "chromatic": list(range(12)),
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}

# MIDI note for keys (octave-agnostic, C=0)
KEY_TO_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

GrooveName = Literal["techno_straight", "house_swing", "jazz_swing_triplet"]


# =============================================================================
# HELPERS
# =============================================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))

def _midi_to_freq(midi: int, a4: float = 440.0) -> float:
    return a4 * (2.0 ** ((midi - 69) / 12.0))

def _pc(midi: int) -> int:
    return midi % 12

def _scale_pitch_classes(key: str, scale_name: str) -> List[int]:
    base = KEY_TO_PC.get(key, 0)
    intervals = SCALES.get(scale_name, SCALES["minor"])
    return sorted([(base + i) % 12 for i in intervals])

def _pick_weighted(rng: np.random.RandomState, items: List[int], weights: List[float]) -> int:
    w = np.array(weights, dtype=np.float64)
    w = np.maximum(1e-9, w)
    w = w / w.sum()
    idx = rng.choice(len(items), p=w)
    return items[int(idx)]

def _nearest_in_set(midi: int, allowed_pcs: List[int], prefer_down: bool = True) -> int:
    """
    Snap midi to nearest pitch class in allowed_pcs (keep octave).
    """
    base_oct = (midi // 12) * 12
    candidates = []
    for pc0 in allowed_pcs:
        # try within same octave
        candidates.append(base_oct + pc0)
        # and neighbor octaves
        candidates.append(base_oct + 12 + pc0)
        candidates.append(base_oct - 12 + pc0)
    candidates = sorted(set(candidates), key=lambda x: (abs(x - midi), (midi - x if prefer_down else x - midi)))
    return candidates[0]


# =============================================================================
# GROOVE PROFILES
# =============================================================================

@dataclass(frozen=True)
class GrooveProfile:
    name: GrooveName
    swing: float  # 0..0.35 range recommended
    triplet: bool
    humanize: float  # 0..1 (timing jitter amount)

GROOVES: Dict[GrooveName, GrooveProfile] = {
    "techno_straight": GrooveProfile("techno_straight", swing=0.0, triplet=False, humanize=0.05),
    "house_swing": GrooveProfile("house_swing", swing=0.16, triplet=False, humanize=0.08),
    "jazz_swing_triplet": GrooveProfile("jazz_swing_triplet", swing=0.22, triplet=True, humanize=0.12),
}

def groove_offsets(steps: int, groove: GrooveName, rng: np.random.RandomState) -> np.ndarray:
    """
    Returns timing offsets in "fractions of a step" (negative/positive).
    Apply later during audio rendering if desired.
    """
    g = GROOVES.get(groove, GROOVES["techno_straight"])
    offs = np.zeros(steps, dtype=np.float32)

    # Swing: push even 16ths later (classic)
    if g.swing > 0:
        for i in range(steps):
            if i % 2 == 1:
                offs[i] += g.swing  # delay

    # Jazz triplet feel: bias 2nd & 3rd subdivision (if you render triplets)
    # We still provide a gentle shape to keep compatibility.
    if g.triplet:
        # create a 3-step wave feel across 12-step blocks
        for i in range(steps):
            m = i % 3
            if m == 1:
                offs[i] += 0.08
            elif m == 2:
                offs[i] -= 0.04

    # Humanize jitter (small)
    jitter = (rng.rand(steps) - 0.5) * 2.0 * g.humanize * 0.1
    offs += jitter.astype(np.float32)

    return offs


# =============================================================================
# PATTERN GENERATOR
# =============================================================================

class PatternGenerator:
    """
    Deterministic Pattern Generator.
    Same seed => same phrases, chords, and mutations.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    # -------------------------------------------------------------------------
    # Legacy API (kept for compatibility)
    # -------------------------------------------------------------------------

    def generate_rhythm(
        self,
        steps: int = 16,
        density: float = 0.5,
        syncopation: float = 0.0,
        force_downbeat: bool = True
    ) -> np.ndarray:
        density = _clamp(density, 0.0, 1.0)
        syncopation = _clamp(syncopation, 0.0, 1.0)

        pattern = np.zeros(steps, dtype=int)

        # Weight strong beats
        weights = np.ones(steps, dtype=np.float32) * 0.5
        for i in range(steps):
            if i % 4 == 0:
                weights[i] += 0.4
            if i % 4 == 2:
                weights[i] += 0.2
            if (i % 2) == 1:
                weights[i] += syncopation * 0.5

        for i in range(steps):
            prob = _clamp(density * float(weights[i]), 0.05, 0.95)
            if self.rng.rand() < prob:
                pattern[i] = 1

        if force_downbeat and steps > 0:
            pattern[0] = 1
        return pattern

    def generate_melody(
        self,
        scale_name: str = "minor",
        root_freq: float = 440.0,
        steps: int = 16,
        complexity: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy melody: returns (freqs, gates).
        """
        gates = self.generate_rhythm(steps, density=_clamp(complexity, 0.0, 1.0), syncopation=0.3)
        freqs = np.zeros(steps, dtype=float)

        scale = SCALES.get(scale_name, SCALES["minor"])
        current_idx = 0

        for i in range(steps):
            if gates[i] == 1:
                jump_range = 1 if complexity < 0.4 else (2 if complexity < 0.7 else 4)
                step = self.rng.randint(-jump_range, jump_range + 1)

                if current_idx > 5:
                    step -= 1
                if current_idx < -2:
                    step += 1

                current_idx += step
                scale_len = len(scale)
                octave = current_idx // scale_len
                note_idx = current_idx % scale_len

                semitones = scale[note_idx] + octave * 12
                freq = root_freq * (2 ** (semitones / 12.0))

                # Acid octave jumps
                if scale_name == "acid" and self.rng.rand() > 0.85:
                    freq *= 2.0

                freqs[i] = float(freq)
            else:
                freqs[i] = 0.0

        return freqs, gates

    def generate_automation(self, steps: int, type: str = "sine") -> np.ndarray:
        t = np.linspace(0, 1, steps, dtype=np.float32)
        if type == "sine":
            speed = int(self.rng.choice([1, 2, 4]))
            phase = float(self.rng.rand()) * 2 * np.pi
            return 0.5 + 0.5 * np.sin(2 * np.pi * speed * t + phase)
        if type == "ramp_up":
            return t
        if type == "ramp_down":
            return 1.0 - t
        if type == "random":
            values = np.zeros(steps, dtype=np.float32)
            val = float(self.rng.rand())
            for i in range(steps):
                val += float(self.rng.uniform(-0.05, 0.05))
                val = _clamp(val, 0.0, 1.0)
                values[i] = val
            return values
        if type == "stepped":
            values = np.zeros(steps, dtype=np.float32)
            hold_len = max(1, steps // 4)
            val = 0.5
            for i in range(steps):
                if i % hold_len == 0:
                    val = float(self.rng.rand())
                values[i] = val
            return values
        return np.full(steps, 0.5, dtype=np.float32)

    # -------------------------------------------------------------------------
    # NEW: PHRASE GENERATION (multi-bar)
    # -------------------------------------------------------------------------

    def generate_rhythm_phrase(
        self,
        steps: int = 64,
        density: float = 0.55,
        syncopation: float = 0.25,
        groove: GrooveName = "techno_straight",
        evolve: float = 0.15,
        motif_bars: int = 1,
        force_downbeat: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (gates, timing_offsets).
        Gates evolve slowly and reuse a motif.
        """
        steps = int(steps)
        density = _clamp(density, 0.0, 1.0)
        syncopation = _clamp(syncopation, 0.0, 1.0)
        evolve = _clamp(evolve, 0.0, 1.0)

        # Build a base motif (1-2 bars)
        motif_steps = max(16, int(motif_bars * 16))
        motif = self.generate_rhythm(motif_steps, density=density, syncopation=syncopation, force_downbeat=force_downbeat)

        # Tile motif across phrase then mutate per bar
        gates = np.tile(motif, int(np.ceil(steps / motif_steps)))[:steps].astype(int)

        bars = max(1, steps // 16)
        for b in range(bars):
            start = b * 16
            end = min(steps, start + 16)

            # Slow evolution: change a few hits
            amt = evolve * (0.02 + 0.06 * (b / max(1, bars - 1)))
            gates[start:end] = self.mutate(gates[start:end], amount=amt)

            # Section accents: slightly reinforce downbeats
            if force_downbeat and start < steps:
                gates[start] = 1

        offs = groove_offsets(steps, groove=groove, rng=self.rng)
        return gates, offs

    def generate_melody_phrase(
        self,
        key: str = "A",
        scale_name: str = "minor",
        steps: int = 64,
        complexity: float = 0.5,
        groove: GrooveName = "techno_straight",
        chord_prog: Optional[List[Dict]] = None,
        resolve: bool = True,
        motif_len: int = 8,
        octave: int = 4,
    ) -> Dict[str, np.ndarray]:
        """
        Returns dict with:
          - midi: int array (0 means rest)
          - freqs: float array (0 means rest)
          - gates: int array
          - offsets: timing offsets
        Melodies prefer chord tones when chord_prog is provided.
        """
        steps = int(steps)
        complexity = _clamp(complexity, 0.0, 1.0)

        # Gates: phrase rhythm
        gates, offs = self.generate_rhythm_phrase(
            steps=steps,
            density=0.35 + complexity * 0.45,
            syncopation=0.15 + complexity * 0.35,
            groove=groove,
            evolve=0.12 + complexity * 0.2,
            motif_bars=1,
            force_downbeat=False
        )

        # Build scale pitch classes and choose register
        allowed_pcs = _scale_pitch_classes(key, scale_name)
        base_midi = 12 * (octave + 1) + KEY_TO_PC.get(key, 9)  # roughly key center

        midi = np.zeros(steps, dtype=np.int32)

        # Create a motif in midi space (short phrase)
        motif_len = max(4, min(int(motif_len), steps))
        motif = []
        cur = base_midi

        # If harmony exists: chord tones drive the motif
        chord_tones_per_step: Optional[List[List[int]]] = None
        if chord_prog is not None and len(chord_prog) > 0:
            chord_tones_per_step = self._expand_chords_to_steps(chord_prog, steps)

        for i in range(motif_len):
            if gates[i] == 0:
                motif.append(0)
                continue

            # Candidate selection: chord tones preferred
            if chord_tones_per_step:
                chord_tones = chord_tones_per_step[i]
                # weight chord tones heavy
                target = _pick_weighted(self.rng, chord_tones, [1.0] * len(chord_tones))
                cur = _nearest_in_set(target, allowed_pcs, prefer_down=True)
            else:
                # scale walk
                step_size = 1 if complexity < 0.4 else (2 if complexity < 0.7 else 4)
                cur += int(self.rng.randint(-step_size, step_size + 1))
                cur = _nearest_in_set(cur, allowed_pcs, prefer_down=True)

            # occasional octave movement (acid-ish)
            if scale_name == "acid" and self.rng.rand() > 0.85:
                cur += 12

            motif.append(int(cur))

        # Paint motif across phrase with slow mutations
        for i in range(steps):
            if gates[i] == 0:
                midi[i] = 0
                continue

            src = motif[i % motif_len]
            if src == 0:
                midi[i] = 0
                continue

            # Slow evolution: nudge by a scale degree sometimes
            if self.rng.rand() < (0.05 + 0.25 * complexity):
                delta = int(self.rng.choice([-2, -1, 1, 2]))
                src = _nearest_in_set(src + delta, allowed_pcs, prefer_down=(delta < 0))

            # Harmony coupling
            if chord_tones_per_step:
                chord_tones = chord_tones_per_step[i]
                if self.rng.rand() < (0.65 - 0.25 * complexity):
                    src = _nearest_in_set(src, [_pc(x) for x in chord_tones], prefer_down=True)

            midi[i] = int(src)

        # Resolve at phrase end (musical closure)
        if resolve and steps > 0:
            last = steps - 1
            # snap last note to tonic if a note exists near the end
            for k in range(last, max(-1, last - 8), -1):
                if midi[k] != 0:
                    tonic_pc = KEY_TO_PC.get(key, 9)
                    midi[k] = _nearest_in_set(midi[k], [tonic_pc], prefer_down=True)
                    break

        freqs = np.zeros(steps, dtype=np.float32)
        for i in range(steps):
            if midi[i] > 0:
                freqs[i] = _midi_to_freq(int(midi[i]))
            else:
                freqs[i] = 0.0

        return {
            "midi": midi,
            "freqs": freqs,
            "gates": gates.astype(np.int32),
            "offsets": offs.astype(np.float32),
        }

    # -------------------------------------------------------------------------
    # NEW: MUTATION SYSTEM
    # -------------------------------------------------------------------------

    def mutate(self, pattern: np.ndarray, amount: float = 0.05) -> np.ndarray:
        """
        Controlled evolution. For gates: flips a small number of steps.
        For pitches: nudges a few values.
        """
        amount = _clamp(amount, 0.0, 1.0)
        out = np.array(pattern, copy=True)

        if out.size == 0 or amount <= 0:
            return out

        n = max(1, int(np.ceil(out.size * amount)))

        # If binary gates -> flip bits
        if np.all((out == 0) | (out == 1)):
            idx = self.rng.choice(out.size, size=n, replace=False)
            out[idx] = 1 - out[idx]
            # keep the grid stable: avoid completely empty pattern
            if out.sum() == 0:
                out[0] = 1
            return out.astype(int)

        # Else treat as continuous/pitch-like -> small jitter
        idx = self.rng.choice(out.size, size=n, replace=False)
        jitter = self.rng.uniform(-1.0, 1.0, size=n)
        out[idx] = out[idx] + jitter
        return out

    # -------------------------------------------------------------------------
    # NEW: KEY-AWARE CHORD PROGRESSION WITH VOICINGS
    # -------------------------------------------------------------------------

    def get_chord_progression(
        self,
        key: str = "A",
        scale_name: str = "minor",
        length: int = 4,
        style: Literal["pop", "house", "jazz"] = "house",
        octave: int = 3,
    ) -> List[Dict]:
        """
        Returns a list of chord dicts with voicings and tone info.

        Each chord:
        {
          "degree": int,
          "root_midi": int,
          "quality": str,
          "tones_midi": [..],
          "tones_pc": [..],
          "name": "Am7" ...
        }
        """
        length = max(1, int(length))
        scale = SCALES.get(scale_name, SCALES["minor"])
        key_pc = KEY_TO_PC.get(key, 9)

        # Degree patterns (index in scale)
        patterns_pop = [
            [0, 5, 3, 4],  # i VI iv v
            [0, 3, 0, 4],  # i iv i v
            [0, 2, 4, 6],  # i III v VII
            [0, 0, 3, 3],  # i i iv iv
        ]

        patterns_house = [
            [0, 3, 5, 3],  # i iv VI iv
            [0, 5, 4, 3],  # i VI v iv
            [0, 2, 3, 4],  # i III iv v
        ]

        # Simplified jazz movement templates (still key-aware)
        patterns_jazz = [
            [1, 4, 0, 4],  # ii V i V (minor-ish with scale mapping)
            [0, 3, 4, 0],  # i iv V i
            [2, 5, 1, 4],  # III VI ii V
        ]

        if style == "jazz":
            chosen = patterns_jazz[self.rng.randint(0, len(patterns_jazz))]
        elif style == "pop":
            chosen = patterns_pop[self.rng.randint(0, len(patterns_pop))]
        else:
            chosen = patterns_house[self.rng.randint(0, len(patterns_house))]

        chosen = (chosen * ((length + 3) // 4))[:length]

        prog: List[Dict] = []
        for deg in chosen:
            deg = int(deg) % len(scale)
            root_semitone = (key_pc + scale[deg]) % 12

            # Choose chord color
            if style == "jazz":
                quality = self.rng.choice(["m7", "7", "maj7", "m9"])
            else:
                quality = self.rng.choice(["triad", "sus2", "sus4", "m7"])

            root_midi = 12 * (octave + 1) + root_semitone

            tones = self._build_voicing(root_midi, quality=quality, scale_name=scale_name)
            # snap tones into the scale if needed (keeps it musical)
            allowed_pcs = _scale_pitch_classes(key, scale_name)
            tones = [ _nearest_in_set(t, allowed_pcs, prefer_down=True) for t in tones ]

            prog.append({
                "degree": deg,
                "root_midi": int(root_midi),
                "quality": str(quality),
                "tones_midi": [int(x) for x in tones],
                "tones_pc": [int(_pc(x)) for x in tones],
                "name": self._chord_name(key, root_semitone, quality),
            })

        return prog

    # -------------------------------------------------------------------------
    # INTERNALS: voicings + chord-to-step expansion
    # -------------------------------------------------------------------------

    def _build_voicing(self, root_midi: int, quality: str, scale_name: str) -> List[int]:
        """
        Return MIDI notes for a chord voicing around root_midi.
        """
        # intervals from root
        if quality == "triad":
            intervals = [0, 3, 7] if scale_name in ("minor", "dorian", "phrygian") else [0, 4, 7]
        elif quality == "sus2":
            intervals = [0, 2, 7]
        elif quality == "sus4":
            intervals = [0, 5, 7]
        elif quality == "m7":
            intervals = [0, 3, 7, 10]
        elif quality == "7":
            intervals = [0, 4, 7, 10]
        elif quality == "maj7":
            intervals = [0, 4, 7, 11]
        elif quality == "m9":
            intervals = [0, 3, 7, 10, 14]
        else:
            intervals = [0, 3, 7, 10]

        tones = [root_midi + i for i in intervals]

        # Voice-leading: small inversion chance
        if self.rng.rand() > 0.6 and len(tones) >= 3:
            # move root up an octave (first inversion-ish)
            tones[0] += 12
            tones = sorted(tones)

        # Spread voicing a bit for piano/jazz
        if self.rng.rand() > 0.7 and len(tones) >= 4:
            tones[1] += 12  # open voicing
            tones = sorted(tones)

        return tones

    def _chord_name(self, key: str, root_pc: int, quality: str) -> str:
        # Basic note names
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root_name = names[int(root_pc)]
        if quality == "triad":
            return root_name
        return f"{root_name}{quality}"

    def _expand_chords_to_steps(self, chord_prog: List[Dict], steps: int) -> List[List[int]]:
        """
        Expand chord progression into chord tones per step.
        Assumes each chord covers equal region of the phrase.
        """
        steps = int(steps)
        per = max(1, steps // max(1, len(chord_prog)))
        expanded: List[List[int]] = []
        for chord in chord_prog:
            tones = chord.get("tones_midi", [])
            for _ in range(per):
                expanded.append(list(tones))
        # pad/truncate
        if len(expanded) < steps:
            expanded += [expanded[-1]] * (steps - len(expanded))
        return expanded[:steps]
