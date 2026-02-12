import assert from 'node:assert/strict';
import { splitHighlightedText } from './evidence.js';

const excerpt = 'Temporal graphs support fast reachability queries over dynamic networks.';
const query = 'reachability in temporal graphs';

const parts = splitHighlightedText(excerpt, query);
const highlighted = parts.filter((p) => p.highlight).map((p) => p.text.toLowerCase());

assert.ok(highlighted.includes('temporal'));
assert.ok(highlighted.includes('graphs'));
assert.ok(highlighted.includes('reachability'));

const zhQuery = '时序图可达性有哪些方法';
const expandedZhQuery = `${zhQuery} reachability temporal graph method`;
const zhParts = splitHighlightedText(excerpt, expandedZhQuery);
const zhHighlighted = zhParts.filter((p) => p.highlight).map((p) => p.text.toLowerCase());

assert.ok(zhHighlighted.length > 0);
assert.ok(zhHighlighted.includes('reachability'));

console.log('PASS evidence highlighting behavior');
