// TTS Worker - Dedicated worker file to avoid dynamic import conflicts
// This file is designed to work properly with Vite's worker bundling system

// Import necessary modules at the top level (static imports)
import { PureKokoroTTS, TextSplitterStream } from "./parrotspeech/pure-kokoro";
import { setVoiceDataUrl } from "./parrotspeech/voices";
import { detectWebGPU } from "./utils";

// Set voice data URL to local files
setVoiceDataUrl("/models/voices");

// Helper function to convert Float32Array audio to Blob
function audioToBlob(audioData: Float32Array, sampleRate: number): Blob {
  // Create WAV file format
  const length = audioData.length;
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  
  // WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, length * 2, true);
  
  // Convert float32 to int16
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }
  
  return new Blob([buffer], { type: 'audio/wav' });
}

// Initialize TTS instance
let tts: PureKokoroTTS | null = null;

// Async initialization function
async function initializeModel() {
  try {
    // Device detection
    const device = (await detectWebGPU()) ? "webgpu" : "wasm";
    self.postMessage({ status: "device", device });

    // Initialize TTS instance
    tts = new PureKokoroTTS();

    // Load the model from local files
    const modelPath = "/models";
    
    // Set up execution providers with fallback
    let executionProviders: string[];
    if (device === "wasm") {
      executionProviders = ["wasm"];
    } else {
      // Try webgpu first, fall back to wasm
      executionProviders = ["webgpu", "wasm"];
    }

    await tts.load(modelPath, {
      executionProviders: executionProviders as any,
    });
    
    console.log("✅ Model loaded successfully in worker");
    self.postMessage({ status: "ready", voices: tts.voices, device });
  } catch (e: any) {
    console.error("❌ Worker initialization failed:", e);
    self.postMessage({ status: "error", error: e.message });
    throw e;
  }
}

// Start initialization
initializeModel();

// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
  if (!tts) {
    self.postMessage({ status: "error", error: "Model not loaded yet" });
    return;
  }

  const { text, voice, speed } = e.data;

  try {
    const streamer = new TextSplitterStream();
    streamer.push(text);
    streamer.close(); // Indicate we won't add more text

    const chunks: { data: Float32Array; sampleRate: number }[] = [];
    
    // Generate audio chunks
    for await (const { text: chunkText, audio } of tts.stream(streamer, { voice, speed })) {
      const audioBlob = audioToBlob(audio.data, audio.sampleRate);
      
      self.postMessage({
        status: "stream",
        chunk: {
          audio: audioBlob,
          text: chunkText,
        },
      });
      chunks.push(audio);
    }

    // Merge chunks
    let finalAudio: Blob | null = null;
    if (chunks.length > 0) {
      const sampleRate = chunks[0].sampleRate;
      const totalLength = chunks.reduce((sum, chunk) => sum + chunk.data.length, 0);
      const mergedWaveform = new Float32Array(totalLength);
      
      let offset = 0;
      for (const chunk of chunks) {
        mergedWaveform.set(chunk.data, offset);
        offset += chunk.data.length;
      }
      
      finalAudio = audioToBlob(mergedWaveform, sampleRate);
    }

    self.postMessage({ status: "complete", audio: finalAudio });
  } catch (error: any) {
    self.postMessage({ status: "error", error: error.message });
  }
});
