"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                           CLEANDICTATE V2.0                                    ║
║                    Production Mission Control for Dictation                    ║
║                                                                                ║
║  Architecture: Brutalist UI + Dual-Model NLP + Split-Core Engine              ║
║  Hardware: NVIDIA RTX 4070 (CUDA + Float16)                                   ║
║  Designer: Senior Python Full-Stack Architect                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝

SYSTEM OVERVIEW:
================
1. GUI Layer: Tkinter "Paper Terminal" - High-Contrast Brutalist Design
2. NLP Brain: Dual-Model Pipeline (Correction → Creation)
3. Engine Core: Split-Mode (Live Stream vs Batch Document)
4. Hardware: GPU-accelerated with thread-safe coordination

DEPENDENCIES:
=============
- tkinter (Standard Library)
- faster_whisper (ASR)
- transformers (NLP)
- pynput (Global Hotkeys + Typing)
- torch (CUDA Backend)
- spacy (POS Tagging)
- pyaudio (Audio Capture)
"""

import numpy as np
import pyaudio
import torch
import threading
import queue
import time
import re
import spacy
import difflib
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController, Key


# =============================================================================
# UTILITY: CONSOLE REDIRECTION TO GUI
# =============================================================================

class ConsoleRedirector:
    """
    Thread-safe stdout/stderr redirector to Tkinter Text widget.
    Captures all print() statements and displays them in the GUI console.
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = []
        self.lock = threading.Lock()
        
    def write(self, message):
        """Write to the console widget (thread-safe)"""
        with self.lock:
            self.buffer.append(message)
            # Schedule GUI update on main thread
            self.text_widget.after(0, self._flush)
    
    def _flush(self):
        """Flush buffer to text widget (runs on main thread)"""
        with self.lock:
            if self.buffer:
                text = ''.join(self.buffer)
                self.buffer.clear()
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)  # Auto-scroll
    
    def flush(self):
        """Required for stdout/stderr interface"""
        pass


# =============================================================================
# STAGE 2 & 3: THE CLEANING ENGINE (Aggressive Mode + Intelligent Stutter Removal)
# =============================================================================

class TextCleaner:
    """
    Hybrid text cleaning engine using regex, spaCy POS tagging, and fuzzy n-gram matching.
    
    Stage 2: Aggressive "Kill List" for Indian English with heavy filler removal.
    Stage 3: Intelligent repetition/stutter removal using fuzzy matching.
    
    Optimized for <15ms latency by disabling unused spaCy components.
    """
    
    # ==========================================================================
    # LAYER 1: THE KILL LIST (Regex-based removal)
    # ==========================================================================
    
    # Universal Fillers (English speech disfluencies)
    FILLER_WORDS_UNIVERSAL = [
        'umm', 'um', 'uhh', 'uh', 'ahh', 'ah',
        'errm', 'erm', 'er', 'hmm', 'hm',
        'huh', 'mhm', 'mm',
        'oops', 'whoops',
        'oh', 'ugh',
    ]
    
    # Indian Context Fillers (Hindi/Hinglish code-switching artifacts)
    FILLER_WORDS_INDIAN = [
        'matlab',           # "I mean" - extremely common filler
        'accha', 'acha',    # "Okay/I see" - acknowledgment filler
        'arrey', 'arey',    # "Hey/Oh" - exclamation filler
        'bas',              # "Just/Stop/That's it" - filler
        'ya', 'yaa',        # "Yeah" - casual agreement
        'na',               # "Right?" - tag question filler
        'haan', 'han',      # "Yes" - filler acknowledgment
        'uff',              # Frustration noise
        'hanji',            # Polite "yes" - usually filler in dictation
        'chalo',            # "Let's go" - transitional filler
        'toh',              # "So" - Hindi connector, often filler
        'kya',              # "What" - sometimes filler
        'dekho',            # "Look/See" - attention filler
        'suno',             # "Listen" - attention filler
        'yaar',             # "Dude/Friend" - casual filler
        'bhai',             # "Brother" - casual address filler
    ]
    
    # ==========================================================================
    # STAGE 3: INTELLIGENT REPETITION REMOVAL (Safe List)
    # ==========================================================================
    
    # Grammatically valid repetitions that should NOT be removed
    SAFE_REPETITIONS = {
        'had had',          # "I had had enough" - past perfect
        'that that',        # "I know that that is true" - demonstrative + conjunction
        'is is',            # Rare but grammatically possible
        'was was',          # Rare but grammatically possible  
        'can can',          # "A can can hold water" - noun + modal (rare)
        'will will',        # Proper noun + modal
        'do do',            # "I do do that sometimes" - emphatic do
        'does does',        # Similar emphatic construction
        'so so',            # "It was so so good" - intentional
    }
    
    # Partial word stutter pattern (catches "I star- started")
    PARTIAL_STUTTER_PATTERN = r'\b(\w+)-\s*(\w+)'
    
    def __init__(self):
        """Initialize the text cleaner with optimized spaCy pipeline"""
        print(f"{Fore.YELLOW}[Cleaner] Loading spaCy model (en_core_web_sm)...{Style.RESET_ALL}")
        
        # Load spaCy with ONLY the tagger for POS tagging - disable parser and NER for speed
        # This is CRITICAL for <15ms latency
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        
        # Compile regex patterns for filler word removal
        all_fillers = self.FILLER_WORDS_UNIVERSAL + self.FILLER_WORDS_INDIAN
        # Create pattern: \b(word1|word2|...)\b with word boundaries, case-insensitive
        filler_pattern = r'(?i)\b(' + '|'.join(re.escape(word) for word in all_fillers) + r')\b'
        self.filler_regex = re.compile(filler_pattern)
        
        # Compile partial stutter pattern (e.g., "star- started")
        self.partial_stutter_regex = re.compile(self.PARTIAL_STUTTER_PATTERN, re.IGNORECASE)
        
        # Words that need POS-based filtering (Layer 2)
        self.pos_filter_words = {'like', 'well', 'basically', 'actually', 'obviously', 'literally', 'right', 'so'}
        
        # "You know" detection
        self.you_know_pattern = re.compile(r'(?i)\byou know\b')
        
        # Fuzzy matching threshold for stutter detection
        self.fuzzy_threshold = 0.85
        
        print(f"{Fore.GREEN}[Cleaner] Ready! (Aggressive Mode + Intelligent Stutter Removal){Style.RESET_ALL}")
    
    def _ghost_normalize(self, text: str) -> str:
        """
        Ghost Normalization: Strip punctuation and lowercase for comparison.
        "No," becomes "no", "Hello!" becomes "hello"
        """
        # Remove all punctuation and lowercase
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        return normalized.strip()
    
    def _is_safe_repetition(self, phrase: str) -> bool:
        """
        Check if a repeated phrase is grammatically valid and should be kept.
        """
        normalized = self._ghost_normalize(phrase)
        return normalized in self.SAFE_REPETITIONS
    
    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """
        Calculate similarity ratio between two strings using difflib.
        Returns a value between 0.0 and 1.0.
        """
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    
    def remove_stutter(self, text: str) -> str:
        """
        Stage 3: Intelligent Repetition Removal using Fuzzy N-Gram Logic.
        
        Algorithm:
        1. Iterate through window sizes N from 3 down to 1 (phrases first, then words)
        2. Compare current window vs previous window with ghost normalization
        3. Use fuzzy matching (>0.85 similarity) to catch partial stutters
        4. Respect safe list for grammatically valid repetitions
        
        Args:
            text: Text after Stage 2 cleaning
            
        Returns:
            Text with intelligent stutter removal applied
        """
        if not text or not text.strip():
            return text
        
        # First, handle partial word stutters like "I star- started"
        # These are hyphenated incomplete words followed by the complete word
        def replace_partial_stutter(match):
            partial = match.group(1)  # "star"
            complete = match.group(2)  # "started"
            # Check if complete word starts with the partial (it's a stutter)
            if complete.lower().startswith(partial.lower()):
                return complete  # Keep only the complete word
            else:
                # Not a stutter, keep both with hyphen
                return match.group(0)
        
        text = self.partial_stutter_regex.sub(replace_partial_stutter, text)
        
        # Tokenize into words while preserving punctuation attachment
        words = text.split()
        
        if len(words) < 2:
            return text
        
        # Process with sliding window - from larger n-grams to smaller
        # This ensures we catch phrase repetitions before word repetitions
        for n in range(3, 0, -1):  # N = 3, 2, 1
            if len(words) < n * 2:
                continue
            
            i = n  # Start from position where we have a previous window
            while i <= len(words) - n:
                # Current window
                current_window = words[i:i + n]
                # Previous window
                previous_window = words[i - n:i]
                
                # Ghost normalize both windows for comparison
                current_normalized = ' '.join(self._ghost_normalize(w) for w in current_window)
                previous_normalized = ' '.join(self._ghost_normalize(w) for w in previous_window)
                
                # Skip empty normalized strings
                if not current_normalized or not previous_normalized:
                    i += 1
                    continue
                
                # Check for match (exact or fuzzy)
                is_exact_match = current_normalized == previous_normalized
                fuzzy_ratio = self._fuzzy_match(current_normalized, previous_normalized)
                is_fuzzy_match = fuzzy_ratio > self.fuzzy_threshold
                
                if is_exact_match or is_fuzzy_match:
                    # Check if this is a safe/grammatically valid repetition
                    # For single words, check if the repeated word pair is in safe list
                    if n == 1:
                        bigram = current_normalized + ' ' + current_normalized
                        if bigram in self.SAFE_REPETITIONS:
                            i += 1
                            continue
                    
                    # For multi-word, check full phrase
                    phrase_to_check = previous_normalized + ' ' + current_normalized
                    if self._is_safe_repetition(phrase_to_check):
                        i += 1
                        continue
                    
                    # Not safe - remove the FIRST occurrence (keep the second which may have more context)
                    # But also strip trailing punctuation from the word before the removed section
                    if i - n >= 0 and i - n + n - 1 < len(words):
                        # Clean punctuation from the last word of first occurrence if removing it
                        last_word_idx = i - 1
                        if last_word_idx >= 0:
                            words[last_word_idx] = re.sub(r'[,;:]+$', '', words[last_word_idx])
                    
                    # Remove the first occurrence (previous window)
                    words = words[:i - n] + words[i:]
                    i = max(n, i - n)  # Reset position
                else:
                    i += 1
        
        result = ' '.join(words)
        
        # Clean up any double spaces created
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Clean orphaned punctuation at start
        result = re.sub(r'^[,;:\s]+', '', result).strip()
        
        return result
    
    def clean(self, text: str) -> str:
        """
        Clean the transcribed text using hybrid approach:
        Layer 1: Regex-based aggressive filler word removal
        Layer 2: POS-based contextual filtering
        Layer 3: Intelligent repetition/stutter removal
        
        Args:
            text: Raw transcribed text from Whisper
            
        Returns:
            Cleaned text with fillers and stutters removed
        """
        if not text or not text.strip():
            return text
        
        # =======================================================================
        # LAYER 1: Regex-based removal (The Kill List)
        # =======================================================================
        
        # Remove "you know" filler phrases first
        text = self.you_know_pattern.sub('', text)
        
        # Remove all filler words from kill list
        text = self.filler_regex.sub('', text)
        
        # Clean up multiple spaces created by removal
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return ""
        
        # =======================================================================
        # LAYER 2: POS-based contextual filtering (spaCy)
        # =======================================================================
        
        doc = self.nlp(text)
        
        cleaned_tokens = []
        skip_next = False
        
        for i, token in enumerate(doc):
            # Skip if marked from previous iteration (for multi-word patterns)
            if skip_next:
                skip_next = False
                continue
            
            word_lower = token.text.lower()
            
            # Skip whitespace tokens
            if token.is_space:
                continue
            
            should_keep = True
            
            # Check words that need POS-based filtering
            if word_lower in self.pos_filter_words:
                pos = token.pos_
                
                if word_lower == 'like':
                    # Delete if INTJ (interjection) or SCONJ at start, keep if VERB
                    if pos == 'INTJ':
                        should_keep = False
                    elif pos in ('ADP', 'SCONJ') and i == 0:
                        should_keep = False
                
                elif word_lower == 'well':
                    # Delete if INTJ, keep if ADV (e.g., "well done") or NOUN (e.g., "water well")
                    if pos == 'INTJ':
                        should_keep = False
                
                elif word_lower in ('basically', 'actually', 'obviously', 'literally'):
                    # Delete if at the start of sentence (index 0)
                    if i == 0:
                        should_keep = False
                
                elif word_lower == 'right':
                    # Delete if INTJ (e.g., "It works, right?")
                    if pos == 'INTJ':
                        should_keep = False
                
                elif word_lower == 'so':
                    # Delete "so" only if at start AND followed by comma or pause
                    if i == 0 and pos in ('ADV', 'INTJ', 'CCONJ'):
                        # Check if next token is comma or if it's a filler "so"
                        if i + 1 < len(doc) and doc[i + 1].text == ',':
                            should_keep = False
                            skip_next = True  # Skip the comma too
                        elif pos == 'INTJ':
                            should_keep = False
            
            # Check for "you know" pattern (POS backup - if regex missed it)
            if word_lower == 'you' and i + 1 < len(doc):
                next_word = doc[i + 1].text.lower()
                if next_word == 'know':
                    next_pos = doc[i + 1].pos_
                    # If "know" is not functioning as main verb, remove both
                    if next_pos in ('VERB',) and (i + 2 >= len(doc) or doc[i + 2].pos_ in ('PUNCT', 'CCONJ', 'SCONJ')):
                        should_keep = False
                        skip_next = True
            
            if should_keep:
                # Preserve original spacing/punctuation
                if token.whitespace_:
                    cleaned_tokens.append(token.text + token.whitespace_)
                else:
                    cleaned_tokens.append(token.text)
        
        # Join and clean up
        result = ''.join(cleaned_tokens).strip()
        
        # =======================================================================
        # LAYER 3: Intelligent Repetition/Stutter Removal (Stage 3)
        # =======================================================================
        
        result = self.remove_stutter(result)
        
        # =======================================================================
        # FINAL CLEANUP
        # =======================================================================
        
        # Fix spacing around punctuation
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        
        # Remove orphaned punctuation at start
        result = re.sub(r'^[,;:\s]+', '', result)
        
        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Capitalize first letter if needed
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        
        return result


# =============================================================================
# STAGE 4: THE GRAMMAR & TONE ENGINE (CoEdit-based)
# =============================================================================

class StyleEngine:
    """
    Dual-Model Cascade Architecture for Grammar & Tone Correction.
    
    Architecture Design Rationale:
    -----------------------------
    Problem: Single models are either "fast but dumb" or "smart but lazy".
    Solution: Decouple CORRECTION from STYLIZATION into a two-stage pipeline.
    
    Model A (The Specialist): vennify/t5-base-grammar-correction (~450MB)
    - Purpose: Pure syntax fixing with surgical precision
    - Runs on: EVERY sentence (grammar errors are universal)
    - Strategy: Greedy decoding (num_beams=1) for speed (~50-80ms)
    - Why this model: Fine-tuned specifically on grammar error datasets,
      outperforms larger general models on pure correction tasks.
    
    Model B (The Generalist): google/flan-t5-large (~1.5GB)
    - Purpose: Semantic restyling (tone transfer)
    - Runs on: ONLY when mode != "Neutral" (conditional execution saves ~200ms)
    - Strategy: Beam search + temperature for creative rewrites
    - Why this model: Strong instruction-following capability, handles
      nuanced style transfer better than specialist models.
    
    Memory Budget (RTX 4070, 8GB VRAM):
    - Model A (float16): ~450MB
    - Model B (float16): ~1.5GB
    - Whisper base.en: ~150MB
    - Total: ~2.1GB (well within 8GB limit)
    
    Latency Profile:
    - Neutral mode: ~50-80ms (Model A only)
    - Style modes: ~250-350ms (Model A + Model B)
    """
    
    # ==========================================================================
    # PROMPT ENGINEERING
    # ==========================================================================
    
    # Model A: Grammar correction prefix (required by vennify/t5-base-grammar-correction)
    GRAMMAR_PREFIX = "grammar: "
    
    # Model B: Style transfer prompts
    # Design philosophy: Use "Rewrite" verbs to force generation, not just copying.
    # Each prompt explicitly states the transformation to prevent lazy pass-through.
    STYLE_PROMPTS = {
        'formal': 'Rewrite the following text in a professional, formal business tone. '
                  'Use proper vocabulary and complete sentences: ',
        'casual': 'Rewrite the following text in a friendly, casual conversational tone. '
                  'Use contractions and relaxed language: ',
        'concise': 'Rewrite the following text to be shorter and more direct. '
                   'Remove unnecessary words while preserving the core meaning: ',
    }
    
    def __init__(self):
        """
        Initialize both models on GPU with float16 precision.
        
        Loading order matters for VRAM fragmentation - load smaller model first.
        """
        print(f"{Fore.YELLOW}[Style] Initializing Dual-Model Cascade Architecture...{Style.RESET_ALL}")
        
        # ======================================================================
        # MODEL A: The Specialist (Grammar Correction)
        # ======================================================================
        print(f"{Fore.YELLOW}[Style] Loading Model A: vennify/t5-base-grammar-correction...{Style.RESET_ALL}")
        
        self.grammar_model_name = "vennify/t5-base-grammar-correction"
        
        # Load tokenizer for grammar model
        self.grammar_tokenizer = AutoTokenizer.from_pretrained(self.grammar_model_name)
        
        # Load grammar model on GPU with float16
        # Using device_map="auto" for automatic CUDA placement
        # use_safetensors=True to avoid PyTorch <2.6 vulnerability check
        self.grammar_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.grammar_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=True
        )
        self.grammar_model.eval()
        
        print(f"{Fore.GREEN}[Style] Model A loaded (GPU float16){Style.RESET_ALL}")
        
        # ======================================================================
        # MODEL B: The Generalist (Style Transfer)
        # ======================================================================
        print(f"{Fore.YELLOW}[Style] Loading Model B: google/flan-t5-large...{Style.RESET_ALL}")
        
        self.style_model_name = "google/flan-t5-large"
        
        # Load tokenizer for style model
        self.style_tokenizer = AutoTokenizer.from_pretrained(self.style_model_name)
        
        # Load style model on GPU with float16
        # use_safetensors=True to avoid PyTorch <2.6 vulnerability check
        self.style_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.style_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=True
        )
        self.style_model.eval()
        
        print(f"{Fore.GREEN}[Style] Model B loaded (GPU float16){Style.RESET_ALL}")
        
        # Warm up both models to eliminate first-inference latency spike
        self._warmup()
        
        print(f"{Fore.GREEN}[Style] Dual-Model Cascade Ready!{Style.RESET_ALL}")
    
    def _warmup(self):
        """
        Run dummy inference on both models to:
        1. Trigger CUDA kernel compilation
        2. Allocate GPU memory pools
        3. Eliminate cold-start latency (~500ms -> ~50ms on first real inference)
        """
        print(f"{Fore.YELLOW}[Style] Warming up models...{Style.RESET_ALL}")
        
        # Warm up Model A (Grammar)
        dummy_grammar = self.GRAMMAR_PREFIX + "hello world"
        inputs_a = self.grammar_tokenizer(dummy_grammar, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.grammar_model.generate(
                **inputs_a,
                max_length=32,
                num_beams=1,  # Greedy - matches production config
            )
        
        # Warm up Model B (Style)
        dummy_style = self.STYLE_PROMPTS['formal'] + "hello world"
        inputs_b = self.style_tokenizer(dummy_style, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.style_model.generate(
                **inputs_b,
                max_length=32,
                num_beams=1,
            )
        
        print(f"{Fore.GREEN}[Style] Warm-up complete{Style.RESET_ALL}")
    
    def _correct_grammar(self, text: str) -> str:
        """
        Stage 1: Pure grammar correction using the specialist model.
        
        Design choices:
        - num_beams=1 (Greedy): Prioritize speed over exploration. Grammar
          correction typically has a single "correct" answer, so beam search
          adds latency without quality improvement.
        - max_length=256: Allow for sentence expansion (e.g., adding missing articles)
        - No temperature/sampling: Deterministic output for consistency
        
        Args:
            text: Input text with potential grammar errors
            
        Returns:
            Grammatically corrected text
        """
        # Construct input with required prefix
        input_text = self.GRAMMAR_PREFIX + text
        
        # Tokenize and move to GPU
        inputs = self.grammar_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to("cuda")
        
        # Generate with greedy decoding (speed priority)
        with torch.no_grad():
            outputs = self.grammar_model.generate(
                **inputs,
                max_length=256,
                num_beams=1,  # Greedy decoding for speed (~50-80ms)
            )
        
        # Decode output
        result = self.grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result.strip()
    
    def _apply_style(self, text: str, mode: str) -> str:
        """
        Stage 2: Semantic restyling using the generalist model.
        
        Design choices:
        - num_beams=4: Explore multiple phrasings for better style adherence.
          4 beams is the sweet spot between quality and latency (~200ms).
        - repetition_penalty=1.2: Prevent the model from copying input verbatim.
          Flan-T5 can be "lazy" and just echo the input; this forces rewrites.
        - temperature=0.7: Moderate creativity for natural-sounding variations.
          Lower values (0.3) sound robotic; higher values (1.0) introduce errors.
        - do_sample=True: Required for temperature to have effect.
        
        Args:
            text: Grammar-corrected text from Stage 1
            mode: One of 'formal', 'casual', 'concise'
            
        Returns:
            Restyled text with applied tone
        """
        # Get the appropriate prompt for the mode
        prompt = self.STYLE_PROMPTS.get(mode.lower())
        if not prompt:
            # Fallback (should not happen if called correctly)
            return text
        
        # Construct input
        input_text = prompt + text
        
        # Tokenize and move to GPU
        inputs = self.style_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to("cuda")
        
        # Generate with creative parameters
        with torch.no_grad():
            outputs = self.style_model.generate(
                **inputs,
                max_length=256,
                num_beams=4,  # Beam search for quality
                repetition_penalty=1.2,  # Force rewrites, prevent lazy copying
                temperature=0.7,  # Moderate creativity
                do_sample=True,  # Enable sampling for temperature to work
                early_stopping=True,
            )
        
        # Decode output
        result = self.style_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result.strip()
    
    def process(self, text: str, mode: str = "neutral") -> str:
        """
        Main entry point: Process text through the dual-model cascade.
        
        Cascade Logic:
        ==============
        1. ALWAYS run Model A (Grammar Correction)
           - Grammar errors exist regardless of desired output style
           - Fast (~50-80ms) so minimal overhead
        
        2. CONDITIONALLY run Model B (Style Transfer)
           - ONLY if mode != "neutral"
           - Saves ~200ms latency for users who just want clean grammar
        
        Pipeline: Raw Input -> [Model A: Grammar] -> [Model B: Style?] -> Output
        
        Args:
            text: Input text (after Stage 1-3 cleaning)
            mode: One of 'neutral', 'formal', 'casual', 'concise'
            
        Returns:
            Grammatically correct, optionally restyled text
        """
        if not text or not text.strip():
            return text
        
        # ======================================================================
        # STAGE 1: Grammar Correction (ALWAYS runs)
        # ======================================================================
        corrected_text = self._correct_grammar(text)
        
        # Safety check: If grammar model returned garbage, use original
        if len(corrected_text.strip()) < len(text.strip()) * 0.2:
            corrected_text = text
        
        # ======================================================================
        # STAGE 2: Style Transfer (CONDITIONAL)
        # ======================================================================
        mode_lower = mode.lower()
        
        if mode_lower == "neutral":
            # Skip Model B entirely - user just wants clean grammar
            # This saves ~200ms latency
            final_text = corrected_text
        else:
            # Apply style transformation
            styled_text = self._apply_style(corrected_text, mode_lower)
            
            # Safety check: If style model returned garbage, use grammar-corrected version
            if len(styled_text.strip()) < len(corrected_text.strip()) * 0.2:
                final_text = corrected_text
            else:
                final_text = styled_text
        
        # ======================================================================
        # FINAL CLEANUP
        # ======================================================================
        
        # Ensure proper capitalization
        if final_text and final_text[0].islower():
            final_text = final_text[0].upper() + final_text[1:]
        
        return final_text.strip()


class ASREngine:
    """
    Real-time Automatic Speech Recognition Engine
    Uses Silero VAD for voice activity detection and Faster Whisper for transcription
    """
    
    # Audio configuration
    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    CHUNK_SIZE = 512     # ~30ms chunks for VAD (512 samples at 16kHz ≈ 32ms)
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    
    # VAD configuration
    SILENCE_THRESHOLD_CHUNKS = 15  # ~500ms of silence (15 * 32ms)
    SPEECH_PAD_CHUNKS = 5          # Padding before speech detection
    
    def __init__(self):
        """Initialize the ASR Engine with GPU optimization"""
        print(f"{Fore.CYAN}[CleanDictate] Initializing ASR Engine...{Style.RESET_ALL}")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! This engine requires an NVIDIA GPU.")
        
        print(f"{Fore.YELLOW}[GPU] CUDA Device: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
        
        # Initialize Faster Whisper with float16 for RTX 4070
        print(f"{Fore.YELLOW}[ASR] Loading Faster Whisper model (base.en)...{Style.RESET_ALL}")
        self.whisper_model = WhisperModel(
            model_size_or_path="base.en",
            device="cuda",
            compute_type="float16",  # float16 for best CUDA compatibility on RTX 4070
        )
        
        # Load Silero VAD model (CPU is fast enough and avoids CUDA library conflicts)
        print(f"{Fore.YELLOW}[VAD] Loading Silero VAD model...{Style.RESET_ALL}")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        # Keep VAD on CPU - it's lightweight and avoids CUDA conflicts
        self.vad_model = self.vad_model.to('cpu')
        
        # Get VAD utility functions
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.vad_utils
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Select microphone device (prefer external/earphone mic)
        self.input_device_index = self._select_microphone()
        
        # Style mode for grammar correction (will be set by _select_tone)
        self.current_mode = "Neutral"  # Default value
        
        # Select tone/style for grammar correction
        self._select_tone()
        
        # Transcription queue for background processing
        self.transcription_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # State variables
        self.is_running = False
        self.is_speaking = False
        self.audio_buffer = []
        self.silence_counter = 0
        self.speech_detected_in_buffer = False
        
        # Prompt for Indian English optimization
        self.initial_prompt = "The following is a transcript of a conversation in Indian English."
        
        # Initialize the text cleaner (Stage 2 & 3)
        self.cleaner = TextCleaner()
        
        # Initialize the style/grammar engine (Stage 4)
        self.style_engine = StyleEngine()
        
        # Warm-up inference to eliminate first-run latency
        self._warmup()
        
        print(f"{Fore.GREEN}[CleanDictate] ASR Engine Ready!{Style.RESET_ALL}\n")
    
    def _warmup(self):
        """Run dummy inference to warm up the models and eliminate first-run latency"""
        print(f"{Fore.YELLOW}[Warm-up] Running warm-up inference...{Style.RESET_ALL}")
        
        # Create 1 second of dummy audio (silence)
        dummy_audio = np.zeros(self.SAMPLE_RATE, dtype=np.float32)
        
        # Warm up Whisper
        segments, _ = self.whisper_model.transcribe(
            dummy_audio,
            beam_size=1,
            language="en",
            initial_prompt=self.initial_prompt,
            vad_filter=False
        )
        # Consume the generator to actually run inference
        _ = list(segments)
        
        # Warm up VAD (on CPU)
        dummy_chunk = torch.zeros(512)
        self.vad_model.reset_states()
        _ = self.vad_model(dummy_chunk, self.SAMPLE_RATE)
        
        print(f"{Fore.GREEN}[Warm-up] Complete!{Style.RESET_ALL}")
    
    def _select_microphone(self) -> int:
        """List available microphones and let user select one (prefers external/USB mic)"""
        print(f"\n{Fore.CYAN}[Audio] Available input devices:{Style.RESET_ALL}")
        
        input_devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info))
                print(f"  {Fore.YELLOW}[{len(input_devices) - 1}]{Style.RESET_ALL} {device_info['name']}")
        
        if not input_devices:
            raise RuntimeError("No input devices found!")
        
        # Let user select
        print(f"\n{Fore.CYAN}Enter the number of your earphone/external microphone: {Style.RESET_ALL}", end="")
        try:
            selection = int(input().strip())
            if 0 <= selection < len(input_devices):
                selected_device = input_devices[selection]
                print(f"{Fore.GREEN}[Audio] Selected: {selected_device[1]['name']}{Style.RESET_ALL}")
                return selected_device[0]
            else:
                print(f"{Fore.YELLOW}[Audio] Invalid selection, using default device{Style.RESET_ALL}")
                return None
        except (ValueError, EOFError):
            print(f"{Fore.YELLOW}[Audio] Using default device{Style.RESET_ALL}")
            return None
    
    def _select_tone(self):
        """Let user select the dictation tone/style for grammar correction"""
        print(f"\n{Fore.CYAN}[Style] Select Dictation Tone:{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}[1]{Style.RESET_ALL} Neutral (Standard English fix)")
        print(f"  {Fore.YELLOW}[2]{Style.RESET_ALL} Formal  (Professional/Business)")
        print(f"  {Fore.YELLOW}[3]{Style.RESET_ALL} Casual  (Conversational/Slang)")
        print(f"  {Fore.YELLOW}[4]{Style.RESET_ALL} Concise (Short/To-the-point)")
        
        tone_map = {
            "1": "Neutral",
            "2": "Formal",
            "3": "Casual",
            "4": "Concise"
        }
        
        print(f"\n{Fore.CYAN}Enter your choice (1-4): {Style.RESET_ALL}", end="")
        try:
            selection = input().strip()
            if selection in tone_map:
                self.current_mode = tone_map[selection]
                print(f"{Fore.GREEN}[Style] Selected: {self.current_mode}{Style.RESET_ALL}")
            else:
                self.current_mode = "Neutral"
                print(f"{Fore.YELLOW}[Style] Invalid selection, using Neutral{Style.RESET_ALL}")
        except (ValueError, EOFError):
            self.current_mode = "Neutral"
            print(f"{Fore.YELLOW}[Style] Using default: Neutral{Style.RESET_ALL}")
    
    def _audio_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw audio bytes to float32 numpy array normalized to [-1, 1]"""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32
    
    def _check_vad(self, audio_chunk: np.ndarray) -> float:
        """
        Check voice activity in an audio chunk using Silero VAD
        Returns speech probability (0.0 to 1.0)
        """
        # Convert to torch tensor (VAD runs on CPU)
        audio_tensor = torch.from_numpy(audio_chunk)
        
        # Get speech probability
        speech_prob = self.vad_model(audio_tensor, self.SAMPLE_RATE).item()
        
        return speech_prob
    
    def _transcription_worker(self):
        """Background thread worker for transcription"""
        while self.is_running:
            try:
                # Wait for audio to transcribe (with timeout to check is_running)
                audio_data = self.transcription_queue.get(timeout=0.1)
                
                if audio_data is None:
                    continue
                
                # Visual indicator: Processing started
                print(f"{Fore.RED}|{Style.RESET_ALL}", end="", flush=True)
                
                start_time = time.time()
                
                # Run transcription with optimized settings
                segments, info = self.whisper_model.transcribe(
                    audio_data,
                    beam_size=1,  # Greedy search for lowest latency
                    language="en",
                    initial_prompt=self.initial_prompt,
                    vad_filter=False,  # We already did VAD
                    word_timestamps=False,  # Disable for speed
                )
                
                # Collect transcription text
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                
                raw_text = " ".join(text_parts).strip()
                
                # --- FILTER LOGIC START ---
                # 1. Ignore Whisper Hallucinations (Standard & Indian Context)
                ignored_phrases = [
                    "The following is a transcript",
                    "conversation in Indian English",
                    "MBC",
                    "Amara.org",
                    "Subtitle",
                    "utf-8",
                    "Thank you for watching",
                    "Subscribe",
                ]
                if any(phrase.lower() in raw_text.lower() for phrase in ignored_phrases):
                    continue
                
                # 2. Ignore garbage/single characters
                if len(raw_text) < 2:
                    continue
                # --- FILTER LOGIC END ---
                
                latency = (time.time() - start_time) * 1000
                
                if raw_text:
                    # Stage 2 & 3: Clean the text (fillers + stutters)
                    clean_start = time.time()
                    clean_text = self.cleaner.clean(raw_text)
                    clean_latency = (time.time() - clean_start) * 1000
                    
                    # Stage 4: Grammar and tone correction
                    style_start = time.time()
                    final_text = self.style_engine.process(clean_text, mode=self.current_mode)
                    style_latency = (time.time() - style_start) * 1000
                    
                    self.results_queue.put({
                        'raw_text': raw_text,
                        'clean_text': clean_text,
                        'final_text': final_text,
                        'asr_latency_ms': latency,
                        'clean_latency_ms': clean_latency,
                        'style_latency_ms': style_latency
                    })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n{Fore.RED}[Error] Transcription failed: {e}{Style.RESET_ALL}")
    
    def _process_buffered_audio(self):
        """Send buffered audio to transcription thread"""
        if len(self.audio_buffer) > 0:
            # Concatenate all buffered chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Only process if we have meaningful audio (> 0.3 seconds)
            min_samples = int(self.SAMPLE_RATE * 0.3)
            if len(audio_data) >= min_samples:
                # Send to transcription queue (non-blocking)
                self.transcription_queue.put(audio_data)
        
        # Reset buffer
        self.audio_buffer = []
        self.speech_detected_in_buffer = False
    
    def start(self):
        """Start the continuous listening loop"""
        print(f"{Fore.CYAN}[CleanDictate] Starting dictation...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[Info] {Fore.GREEN}• = Speaking{Style.RESET_ALL} | {Fore.RED}| = Processing{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[Info] Press Ctrl+C to stop{Style.RESET_ALL}\n")
        
        self.is_running = True
        
        # Start transcription worker thread
        self.transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.transcription_thread.start()
        
        # Reset VAD state
        self.vad_model.reset_states()
        
        # Open microphone stream
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.CHUNK_SIZE
        )
        
        print(f"{Fore.GREEN}[Listening...]{Style.RESET_ALL} ", end="", flush=True)
        
        try:
            while self.is_running:
                # Read audio chunk from microphone
                audio_bytes = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = self._audio_to_float32(audio_bytes)
                
                # Check for speech activity
                speech_prob = self._check_vad(audio_chunk)
                is_speech = speech_prob > 0.5
                
                if is_speech:
                    # Speech detected
                    if not self.is_speaking:
                        # Speech just started
                        self.is_speaking = True
                        print(f"{Fore.GREEN}•{Style.RESET_ALL}", end="", flush=True)
                    
                    self.speech_detected_in_buffer = True
                    self.audio_buffer.append(audio_chunk)
                    self.silence_counter = 0
                    
                else:
                    # Silence detected
                    if self.is_speaking:
                        # We were speaking, now we have silence
                        self.audio_buffer.append(audio_chunk)  # Include trailing silence
                        self.silence_counter += 1
                        
                        # Check if silence threshold reached (~500ms)
                        if self.silence_counter >= self.SILENCE_THRESHOLD_CHUNKS:
                            # End of speech segment
                            self.is_speaking = False
                            
                            if self.speech_detected_in_buffer:
                                # Process the buffered audio
                                self._process_buffered_audio()
                            else:
                                # Reset buffer without processing
                                self.audio_buffer = []
                            
                            self.silence_counter = 0
                    else:
                        # Keep a small rolling buffer for pre-speech context
                        self.audio_buffer.append(audio_chunk)
                        if len(self.audio_buffer) > self.SPEECH_PAD_CHUNKS:
                            self.audio_buffer.pop(0)
                
                # Check for transcription results
                try:
                    result = self.results_queue.get_nowait()
                    print(f"\n{Fore.YELLOW}[Raw]:   {result['raw_text']}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}[Clean]: {result['clean_text']}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}[Final]: {result['final_text']}{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}   [ASR: {result['asr_latency_ms']:.0f}ms | Clean: {result['clean_latency_ms']:.1f}ms | Style: {result['style_latency_ms']:.0f}ms | Mode: {self.current_mode}]{Style.RESET_ALL}")
                    print(f"\n{Fore.GREEN}[Listening...]{Style.RESET_ALL} ", end="", flush=True)
                except queue.Empty:
                    pass
                    
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}[CleanDictate] Stopping...{Style.RESET_ALL}")
        finally:
            self.stop()
            stream.stop_stream()
            stream.close()
    
    def stop(self):
        """Stop the ASR engine"""
        self.is_running = False
        # Signal transcription thread to stop
        self.transcription_queue.put(None)
        print(f"{Fore.GREEN}[CleanDictate] Stopped.{Style.RESET_ALL}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'audio'):
            self.audio.terminate()


def main():
    """Main entry point"""
    print(f"""
{Fore.CYAN}╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   {Fore.WHITE}CleanDictate - Real-time Dictation Engine{Fore.CYAN}                   ║
║   {Fore.YELLOW}Optimized for Indian English{Fore.CYAN}                               ║
║   {Fore.GREEN}Stage 1: ASR → 2: Cleaner → 3: Stutter → 4: Style{Fore.CYAN}           ║
║   {Fore.MAGENTA}Grammar/Tone: flan-t5-large (GPU float16){Fore.CYAN}                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
    
    # Initialize and start the engine
    engine = ASREngine()
    engine.start()


if __name__ == "__main__":
    main()
