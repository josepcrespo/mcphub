import { useEffect, useRef } from 'react';
import { useToast } from '@/contexts/ToastContext';
import { logStreamManager } from '@/services/logService';

const EMBED_SYNC_ERROR_MARKER = '[EMBED_SYNC_ERROR]';
const TOAST_DURATION_MS = 8000;
const TOAST_DEDUPE_WINDOW_MS = 30000;

const EmbeddingSyncAlertListener = () => {
  const { showToast } = useToast();
  const lastToastMessageRef = useRef<string>('');
  const lastToastAtRef = useRef<number>(0);

  useEffect(() => {
    const shouldSuppressToast = (message: string): boolean => {
      const now = Date.now();
      const isDuplicate =
        message === lastToastMessageRef.current &&
        now - lastToastAtRef.current < TOAST_DEDUPE_WINDOW_MS;

      if (isDuplicate) {
        return true;
      }

      lastToastMessageRef.current = message;
      lastToastAtRef.current = now;
      return false;
    };

    const handleEmbeddingSyncError = (rawMessage: string) => {
      const cleanedMessage = rawMessage.replace(EMBED_SYNC_ERROR_MARKER, '').trim();
      const toastMessage =
        cleanedMessage ||
        'Embedding synchronization failed. Check smart routing logs and provider rate limits.';

      if (shouldSuppressToast(toastMessage)) {
        return;
      }

      showToast(toastMessage, 'error', TOAST_DURATION_MS);
    };

    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data?.type !== 'log') return;
        const message = String(data?.log?.message || '');
        if (!message.includes(EMBED_SYNC_ERROR_MARKER)) return;
        handleEmbeddingSyncError(message);
      } catch {
        // Ignore malformed stream messages.
      }
    };

    return logStreamManager.subscribe(handleMessage);
  }, [showToast]);

  return null;
};

export default EmbeddingSyncAlertListener;
