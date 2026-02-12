import assert from 'node:assert/strict';
import { showcaseContent } from './projectShowcase.js';

assert.equal(typeof showcaseContent.githubUrl, 'string');
assert.ok(showcaseContent.githubUrl.startsWith('https://github.com/'));

assert.ok(showcaseContent.zh.title.includes('IndexGPT'));
assert.ok(showcaseContent.en.subtitle.toLowerCase().includes('open-source'));

assert.ok(showcaseContent.zh.features.length >= 3);
assert.ok(showcaseContent.en.features.length >= 3);

console.log('PASS project showcase content');

