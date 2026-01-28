/**
 * Masks a string, allowing customization of visible characters and masking character.
 *
 * @param {string} text - The input string to be masked.
 * @param {number} visibleStart - Number of characters to remain visible at the start.
 * @param {number} visibleEnd - Number of characters to remain visible at the end.
 * @param {string} maskChar - The character used for masking the hidden part of the string.
 * @returns {string} - The masked string.
 */
export const maskString = (
  text: string,
  visibleStart: number = 4,
  visibleEnd: number = 4,
  maskChar: string = '*'
): string => {
  if (text.length <= visibleStart + visibleEnd) {
    return maskChar.repeat(text.length);
  }

  const start = text.slice(0, visibleStart);
  const end = text.slice(-visibleEnd);
  const maskedSection = maskChar.repeat(text.length - visibleStart - visibleEnd);

  return start + maskedSection + end;
};
