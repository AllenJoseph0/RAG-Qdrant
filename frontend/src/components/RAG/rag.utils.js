// Utility functions for RAG components
export const isLocal =
    window.location.hostname === 'localhost' ||
    window.location.hostname.startsWith('192.168.');

export const RAG_BACKEND_URL = isLocal
    ? 'http://192.168.18.15:8351'
    : 'http://61.2.142.91:8351';

export function generateUUID() {
    var d = new Date().getTime();
    var d2 = ((typeof performance !== 'undefined') && performance.now && (performance.now() * 1000)) || 0;
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        var r = Math.random() * 16;
        if (d > 0) {
            r = (d + r) % 16 | 0;
            d = Math.floor(d / 16);
        } else {
            r = (d2 + r) % 16 | 0;
            d2 = Math.floor(d2 / 16);
        }
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

export function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function formatDate(timestamp) {
    if (!timestamp) return 'N/A';
    try {
        return new Date(timestamp).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch (e) {
        return 'Invalid Date';
    }
}

export function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    try {
        return new Date(timestamp).toLocaleTimeString(undefined, {
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (e) {
        return 'Invalid Time';
    }
}
