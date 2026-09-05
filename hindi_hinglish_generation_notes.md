# Hindi/Hinglish Track — Generation Notes

Working notes from Phase 4, Prompt 4.3 (XTTS-v2 batch clone generation). These feed directly
into the dataset card (Prompt 4.5) and should be reported honestly rather than smoothed over.

Date: 2026-09-05

---

## Dataset composition

| | Count | Notes |
|---|---|---|
| Speakers | 3 | byaquta, mahato, soumya |
| Real clips | 75 | 25 scripted sentences per speaker, recorded at 48 kHz, downsampled to 16 kHz mono |
| Reference clips | 3 | One per speaker, 10.2–12.5 s, unscripted natural speech, used for XTTS-v2 conditioning |
| Synthetic clips | 75 | One matched clone per real clip — same speaker, same sentence |
| Total | 150 | |

Sentence categories per speaker: 10 neutral (casual code-switched), 10 scam-pattern, 5 control
(legitimate phone-call style).

Consent: all three speakers gave explicit written consent to both recording and cloning, obtained
before any clone was generated. Consent status is recorded per speaker in
`data/metadata/hindi_hinglish_real.csv` (`consent_confirmed` column).

---

## Issue 1 — Sentence 24 failed for every speaker (`num2words` has no Hindi support)

Sentence 24 as originally written was:

> Hi, this is a reminder call for your appointment tomorrow at 10 AM.

This failed synthesis for all three speakers, with an empty error message that only became
diagnosable after bypassing the wrapper and printing the full traceback. Root cause:

```
TTS/tts/layers/xtts/tokenizer.py -> expand_numbers_multilingual -> _expand_number
  -> num2words(int(m.group(0)), lang="hi")
  -> NotImplementedError
```

XTTS-v2's text preprocessing expands numerals to words before tokenizing. `num2words` does not
implement Hindi, so **any** sentence containing a bare digit synthesized with `language="hi"`
raises `NotImplementedError`. This is not specific to sentence 24, the speaker, or the reference
clip.

**Workaround applied:** the digit was spelled out — "at ten AM" instead of "at 10 AM". No other
change. The three affected clips were regenerated individually and appended to
`hindi_hinglish_synthetic.csv`.

This is a text-level substitution, not an audio-quality intervention. The real recordings still
use the original spoken "10 AM"; only the text fed to XTTS-v2 was altered.

---

## Issue 2 — Generation is stochastic, and default settings produced unintelligible output

With XTTS-v2's default sampling parameters, the same sentence, same reference clip, and same
settings produced wildly varying output quality run to run. Measured directly: three consecutive
generations of one identical pure-Hindi sentence yielded one intelligible clip and two that did
not resemble any human language.

Two fixes were applied to `src/voxguard/synth/xtts_clone.py`:

1. **`split_sentences=True`** was missing from the `tts_to_file()` call. Adding it produced a
   clear, directly A/B-tested improvement on longer multi-clause sentences, which had previously
   run together into an unparseable stream.

2. **Lower-randomness generation parameters plus retry-on-validation-failure:**
   - `temperature=0.3` (down from XTTS-v2's default of roughly 0.65–0.75)
   - `repetition_penalty=10.0`
   - `length_penalty=1.0`
   - `max_retries=3`, where output duration is checked against a rough expectation derived from
     input text length; anything outside that range triggers regeneration, with the last attempt
     accepted and logged as a warning if all retries fail validation

Lower temperature trades some naturalness for run-to-run consistency. That is the right trade
here, because the output is dataset material rather than a single showcase clip.

---

## Issue 3 — Final quality, stated plainly

After both fixes, across a listening pass spanning all three speakers and a mix of pure-Hindi,
heavily code-switched, and control sentences:

- **Mostly-Hindi sentences:** broadly intelligible.
- **Code-switched (Hinglish) sentences:** roughly 60% intelligible. English words embedded in
  Hindi sentences render with noticeably degraded prosody — this matches the known limitation
  that XTTS-v2 conditions on a single `language` token per call and is not designed for
  intra-sentence code-switching.
- **Accent:** generated speech carries prosodic characteristics that do not fully match the
  source speakers' regional accent. XTTS-v2 captures timbre from the ~10 s reference but its
  Hindi prosody model reflects its own training distribution.
- **Occasional artifacts:** light static on some clips.

**No clips were cherry-picked.** Every generated clip was retained, including awkward ones,
to avoid biasing evaluation toward easy examples. For a detector-training corpus this is the
correct choice: audible synthesis artifacts are genuine properties of real cloned audio, and a
detector validated only on flawless clones would be the weaker result.

A longer reference clip (15–30 s rather than ~10 s) would likely improve conditioning quality.
This was not pursued, as it would require re-recording all three speakers and the current
quality is sufficient for the detector-training purpose.

---

## Suggested dataset card wording

> Synthetic clips were generated with Coqui XTTS-v2 (`temperature=0.3`,
> `repetition_penalty=10.0`, `split_sentences=True`, with retry-on-validation-failure) from
> 10–12 second reference clips, one per speaker. Output quality is intelligible but imperfect:
> code-switched English words within Hinglish sentences render with degraded prosody, and
> generated speech carries accent characteristics not fully matching the source speaker. This
> reflects a genuine limitation of XTTS-v2 for code-switched Indic speech rather than a pipeline
> defect. Clips were retained as-is rather than cherry-picked, to avoid biasing evaluation
> toward easier examples. One sentence containing a numeral required the digit to be spelled out,
> as XTTS-v2's Hindi text normalization raises `NotImplementedError` on any numeric token.

---

## Licensing note (carry forward)

Coqui XTTS-v2 is CPML-licensed: free for personal, research, and non-commercial use. This covers
SIH hackathon use. If VoxGuard is ever taken commercial, the synthetic portion of this track
would need regenerating with a differently-licensed model — AI4Bharat Indic TTS is the
already-identified fallback.
