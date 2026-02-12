import assert from 'node:assert/strict';
import { shouldAutoScroll } from './chatScroll.js';

assert.equal(
  shouldAutoScroll({ scrollTop: 700, scrollHeight: 1000, clientHeight: 260 }),
  true,
  'should auto scroll when user is near bottom'
);

assert.equal(
  shouldAutoScroll({ scrollTop: 300, scrollHeight: 1000, clientHeight: 260 }),
  false,
  'should not auto scroll when user scrolled away from bottom'
);

assert.equal(
  shouldAutoScroll({ scrollTop: 0, scrollHeight: 0, clientHeight: 0 }),
  true,
  'should default to auto scroll when no scrollable content exists'
);

console.log('PASS chat scroll behavior');
