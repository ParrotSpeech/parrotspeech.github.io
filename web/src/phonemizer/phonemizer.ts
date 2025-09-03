import Module from "./espeakng.worker";

// Type definitions for the eSpeak-NG module
interface ESpeakNGLanguage {
  name: string;
  priority: number;
}

interface ESpeakNGVoice {
  name: string;
  identifier: string;
  languages: ESpeakNGLanguage[];
}

interface ESpeakNGWorker {
  list_voices(): ESpeakNGVoice[];
  set_voice(language: string): void;
  synthesize_ipa(text: string): { ipa?: string };
}

interface ESpeakNGModule {
  calledRun?: boolean;
  onRuntimeInitialized?: () => void;
  eSpeakNGWorker: new () => ESpeakNGWorker;
}

const workerPromise = new Promise<ESpeakNGWorker>((resolve) => {
  const module = Module as ESpeakNGModule;
  if (module.calledRun) {
    resolve(new module.eSpeakNGWorker());
  } else {
    module.onRuntimeInitialized = () => resolve(new module.eSpeakNGWorker());
  }
});

const SUPPORTED_LANGUAGES: string[] = [
  "en", // English
];

interface InitCache {
  voices: ESpeakNGVoice[];
  identifiers: Set<string>;
}

const initCache: Promise<InitCache> = workerPromise.then((worker) => {
  const voices = worker
    .list_voices()
    .map(({ name, identifier, languages }) => ({
      name,
      identifier,
      languages: languages.filter((lang) =>
        SUPPORTED_LANGUAGES.includes(lang.name.split("-")[0]),
      ),
    }))
    .filter((voice) => voice.languages.length > 0);

  // Generate list of supported language identifiers:
  const identifiers = new Set<string>();
  for (const voice of voices) {
    identifiers.add(voice.identifier);
    for (const lang of voice.languages) {
      identifiers.add(lang.name);
    }
  }

  return { voices, identifiers };
});

/**
 * List the available voices for the specified language.
 * @param language The language identifier (optional)
 * @returns A list of available voices
 */
export const list_voices = async (language?: string): Promise<ESpeakNGVoice[]> => {
  const { voices } = await initCache;
  if (!language) return voices;
  const base = language.split("-")[0];
  return voices.filter((voice) =>
    voice.languages.some(
      (lang) => lang.name === base || lang.name.startsWith(base + "-"),
    ),
  );
};

/**
 * Multilingual text to phonemes converter
 *
 * @param text The input text
 * @param language The language identifier (defaults to "en-us")
 * @returns A phonemized version of the input
 */
export const phonemize = async (text: string, language: string = "en-us"): Promise<string[]> => {
  const worker = await workerPromise;

  const { identifiers } = await initCache;
  if (!identifiers.has(language)) {
    throw new Error(
      `Invalid language identifier: "${language}". Should be one of: ${Array.from(identifiers).sort().join(", ")}.`,
    );
  }
  worker.set_voice(language);

  return (
    worker
      .synthesize_ipa(text)
      .ipa?.split("\n")
      .filter((x) => x.length > 0) ?? []
  );
};