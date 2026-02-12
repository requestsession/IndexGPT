export function shouldAutoScroll(scrollContainer, thresholdPx = 64) {
  if (!scrollContainer) return true;

  const scrollTop = Number(scrollContainer.scrollTop || 0);
  const scrollHeight = Number(scrollContainer.scrollHeight || 0);
  const clientHeight = Number(scrollContainer.clientHeight || 0);

  if (scrollHeight <= clientHeight) {
    return true;
  }

  const distanceToBottom = scrollHeight - (scrollTop + clientHeight);
  return distanceToBottom <= thresholdPx;
}
