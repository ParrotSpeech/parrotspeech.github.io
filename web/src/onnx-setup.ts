/**
 * ONNX Runtime backend handler inspired by @huggingface/transformers
 * This file handles ONNX Runtime imports and configuration for both main and worker contexts
 */

import * as ONNX_WEB from 'onnxruntime-web';

export { Tensor } from 'onnxruntime-common';

/** @type {Record<string, any>} */
const DEVICE_TO_EXECUTION_PROVIDER_MAPPING: Record<string, any> = Object.freeze({
    auto: null, // Auto-detect based on device and environment
    gpu: null, // Auto-detect GPU
    cpu: 'cpu', // CPU
    wasm: 'wasm', // WebAssembly
    webgpu: 'webgpu', // WebGPU
});

/** 
 * The list of supported devices, sorted by priority/performance.
 */
const supportedDevices: string[] = [];

/** @type {any[]} */
let defaultDevices: any[];
let ONNX: typeof ONNX_WEB;

// In browser environment, use onnxruntime-web
ONNX = ONNX_WEB;

// Check for WebGPU availability
const IS_WEBGPU_AVAILABLE = typeof navigator !== 'undefined' && 'gpu' in navigator;

if (IS_WEBGPU_AVAILABLE) {
    supportedDevices.push('webgpu');
}

supportedDevices.push('wasm');
defaultDevices = ['wasm'];

// @ts-ignore
const InferenceSession = ONNX.InferenceSession;

/**
 * Map a device to the execution providers to use for the given device.
 */
export function deviceToExecutionProviders(device: string | null = null): any[] {
    // Use the default execution providers if the user hasn't specified anything
    if (!device) return defaultDevices;

    // Handle overloaded cases
    switch (device) {
        case "auto":
            return supportedDevices;
        case "gpu":
            return supportedDevices.filter(x =>
                ["webgpu"].includes(x),
            );
    }

    if (supportedDevices.includes(device)) {
        return [DEVICE_TO_EXECUTION_PROVIDER_MAPPING[device] ?? device];
    }

    throw new Error(`Unsupported device: "${device}". Should be one of: ${supportedDevices.join(', ')}.`)
}

/**
 * To prevent multiple calls to `initWasm()`, we store the first call in a Promise
 * that is resolved when the first InferenceSession is created.
 */
let wasmInitPromise: Promise<any> | null = null;

/**
 * Create an ONNX inference session.
 */
export async function createInferenceSession(
    buffer: Uint8Array, 
    session_options: any, 
    session_config: any = {}
): Promise<any> {
    if (wasmInitPromise) {
        // A previous session has already initialized the WASM runtime
        // so we wait for it to resolve before creating this new session.
        await wasmInitPromise;
    }

    const sessionPromise = InferenceSession.create(buffer, session_options);
    wasmInitPromise ??= sessionPromise;
    const session = await sessionPromise;
    (session as any).config = session_config;
    return session;
}

/**
 * Check if an object is an ONNX tensor.
 */
export function isONNXTensor(x: any): boolean {
    return x instanceof ONNX.Tensor;
}

// Configure ONNX Runtime environment
const ONNX_ENV = ONNX?.env;

if (ONNX_ENV?.wasm) {
    // Set path to wasm files - use local files
    ONNX_ENV.wasm.wasmPaths = '/';
    
    // Disable proxy to avoid freezing UI
    ONNX_ENV.wasm.proxy = false;

    // Check if cross-origin isolated for threading
    if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) {
        ONNX_ENV.wasm.numThreads = 1;
    } else {
        ONNX_ENV.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
    }
}

if (ONNX_ENV?.webgpu) {
    ONNX_ENV.webgpu.powerPreference = 'high-performance';
}

// Set log level to reduce noise
if (ONNX_ENV) {
    ONNX_ENV.logLevel = 'error';
}

/**
 * Initialize ONNX Runtime environment
 */
export function setupOnnxRuntime(): typeof ONNX {
    return ONNX;
}

/**
 * Check if ONNX's WASM backend is being proxied.
 */
export function isONNXProxy(): boolean {
    return ONNX_ENV?.wasm?.proxy || false;
}

// Export the configured ONNX runtime
export { ONNX as ort };
export { InferenceSession };
export { supportedDevices, defaultDevices };
export { ONNX_ENV as env };
