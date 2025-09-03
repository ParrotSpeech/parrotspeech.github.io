import * as ort from 'onnxruntime-web';
import { phonemize } from "./phonemize";
import { TextSplitterStream } from "./splitter";
import { getVoiceData, VOICES } from "./voices";

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

// Configure ONNX Runtime for browser
ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = 1;

/**
 * Pure client-side TTS using ONNX Runtime Web
 * No dependency on Hugging Face Transformers
 */
export class PureKokoroTTS {
  private session: ort.InferenceSession | null = null;
  private tokenizer: any = null;

  constructor() {
    // Constructor is empty - models loaded via load() method
  }

  /**
   * Load model, tokenizer and config from local files
   * @param {string} modelPath - Path to the model directory in public folder
   * @param {Object} options - Loading options
   */
  async load(modelPath: string, options: {
    executionProviders?: ('wasm' | 'webgl' | 'webgpu')[];
    wasmPaths?: string;
  } = {}) {
    const { 
      executionProviders = ['wasm'], 
      wasmPaths = '/' 
    } = options;

    try {
      // Set WASM paths
      ort.env.wasm.wasmPaths = wasmPaths;

      // Load config
      console.log('Loading config...');
      const configResponse = await fetch(`${modelPath}/config.json`);
      const config = await configResponse.json();
      console.log('Config loaded:', config);

      // Load tokenizer
      console.log('Loading tokenizer...');
      const tokenizerResponse = await fetch(`${modelPath}/tokenizer.json`);
      this.tokenizer = await tokenizerResponse.json();
      console.log("tokenizer", this.tokenizer);

      // Load ONNX model
      console.log('Loading ONNX model...');
      const modelUrl = `${modelPath}/onnx/model_q8f16.onnx`;
      console.log("modelUrl", modelUrl)
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: executionProviders,
      });

      console.log('✅ Model loaded successfully!');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);

    } catch (error) {
      console.error('❌ Failed to load model:', error);
      throw error;
    }
  }

  /**
   * Simple tokenizer implementation
   * This is a basic implementation - you might need to adjust based on your tokenizer
   */
  private tokenize(phonemes: string): number[] {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not loaded');
    }

    // Simple character-to-ID mapping (this is simplified)
    const vocab = this.tokenizer.model?.vocab || {};
    const tokens: number[] = [];
    
    // Add special tokens
    const bosToken = vocab['<s>'] || 1;
    const eosToken = vocab['</s>'] || 2;
    const unkToken = vocab['<unk>'] || 0;
    
    tokens.push(bosToken);
    
    // Convert phonemes to tokens
    for (const char of phonemes) {
      const tokenId = vocab[char] || unkToken;
      tokens.push(tokenId);
    }
    
    tokens.push(eosToken);
    
    return tokens;
  }

  /**
   * Create ONNX tensor from array
   */
  private createTensor(data: number[] | Float32Array, dims: number[], type: 'int64' | 'float32' = 'float32'): ort.Tensor {
    if (type === 'int64') {
      const numArray = Array.isArray(data) ? data : Array.from(data);
      const int64Data = new BigInt64Array(numArray.map(x => BigInt(x)));
      return new ort.Tensor('int64', int64Data, dims);
    } else {
      const float32Data = data instanceof Float32Array ? data : new Float32Array(data);
      return new ort.Tensor('float32', float32Data, dims);
    }
  }

  get voices() {
    return VOICES;
  }

  list_voices() {
    console.table(VOICES);
  }

  _validate_voice(voice: keyof typeof VOICES) {
    if (!VOICES.hasOwnProperty(voice)) {
      console.error(`Voice "${voice}" not found. Available voices:`);
      console.table(VOICES);
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }
    const language = voice.at(0) as "a" | "b"; // "a" or "b"
    return language;
  }

  /**
   * Generate audio from text using pure ONNX inference
   */
  async generate(text: string, { voice = "af_heart" as keyof typeof VOICES, speed = 1 } = {}) {
    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.');
    }

    const language = this._validate_voice(voice);

    // Get phonemes
    const phonemes = await phonemize(text, language);
    console.log('Phonemes:', phonemes);

    // Tokenize
    const tokens = this.tokenize(phonemes);
    console.log('Tokens:', tokens);

    // Create input tensors
    const inputIds = this.createTensor(tokens, [1, tokens.length], 'int64');
    
    // Load voice style
    const numTokens = Math.min(Math.max(tokens.length - 2, 0), 509);
    const voiceData = await getVoiceData(voice);
    const offset = numTokens * STYLE_DIM;
    const styleData = voiceData.slice(offset, offset + STYLE_DIM);
    const styleTensor = this.createTensor(Array.from(styleData), [1, STYLE_DIM]);
    
    // Speed tensor
    const speedTensor = this.createTensor([speed], [1]);

    // Prepare inputs
    const feeds: Record<string, ort.Tensor> = {
      'input_ids': inputIds,
      'style': styleTensor,
      'speed': speedTensor
    };

    console.log('Running inference...');
    
    // Run inference
    const results = await this.session.run(feeds);
    
    // Get waveform output
    const waveformTensor = results['waveform'];
    if (!waveformTensor) {
      throw new Error('No waveform output found');
    }

    const waveformData = waveformTensor.data as Float32Array;
    
    console.log(`✅ Generated audio: ${waveformData.length} samples at ${SAMPLE_RATE}Hz`);
    
    return {
      data: waveformData,
      sampleRate: SAMPLE_RATE
    };
  }

  /**
   * Generate audio from text in streaming fashion
   */
  async *stream(text: string | TextSplitterStream, { 
    voice = "af_heart" as keyof typeof VOICES, 
    speed = 1, 
    split_pattern = null 
  }: {
    voice?: keyof typeof VOICES;
    speed?: number;
    split_pattern?: RegExp | null;
  } = {}) {
    const language = this._validate_voice(voice);

    let splitter: TextSplitterStream;
    if (text instanceof TextSplitterStream) {
      splitter = text;
    } else if (typeof text === "string") {
      splitter = new TextSplitterStream();
      const chunks = split_pattern
        ? text
          .split(split_pattern)
          .map((chunk) => chunk.trim())
          .filter((chunk) => chunk.length > 0)
        : [text];
      splitter.push(...chunks);
    } else {
      throw new Error("Invalid input type. Expected string or TextSplitterStream.");
    }

    for await (const sentence of splitter) {
      const phonemes = await phonemize(sentence, language);
      const audio = await this.generate(sentence, { voice, speed });
      yield { text: sentence, phonemes, audio };
    }
  }

  /**
   * Dispose of resources
   */
  async dispose() {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }
}

// Helper function to convert audio data to playable format
export function createAudioBuffer(audioData: { data: Float32Array; sampleRate: number }, audioContext: AudioContext): AudioBuffer {
  const buffer = audioContext.createBuffer(1, audioData.data.length, audioData.sampleRate);
  const channelData = new Float32Array(audioData.data);
  buffer.copyToChannel(channelData, 0);
  return buffer;
}

// Helper function to play audio
export async function playAudio(audioData: { data: Float32Array; sampleRate: number }): Promise<void> {
  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
  const buffer = createAudioBuffer(audioData, audioContext);
  
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  
  return new Promise((resolve) => {
    source.onended = () => resolve();
    source.start();
  });
}

export { TextSplitterStream };
