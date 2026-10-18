import { describe, expect, it } from 'vitest';
import { parseAskInput, parseFeedback, parseLibraryInput, uploadExtension } from './validation';
import { sanitizeFilename } from './volumes';

describe('parseLibraryInput', () => {
  it('fills create defaults and normalizes the name', () => {
    expect(parseLibraryInput({ name: '  My   papers ' }, [], false)).toEqual({
      ok: true, value: { name: 'My papers', visibility: 'private', documents_language: 'english' },
    });
  });

  it('only allows sharing with my own groups', () => {
    expect(parseLibraryInput({ name: 'x', visibility: 'group', group_name: 'lab' }, ['lab'], false).ok).toBe(true);
    expect(parseLibraryInput({ name: 'x', visibility: 'group', group_name: 'admins' }, ['lab'], false).ok).toBe(false);
  });

  it('rejects bad values and language changes on update', () => {
    expect(parseLibraryInput({ name: '' }, [], false).ok).toBe(false);
    expect(parseLibraryInput({ name: 'x', visibility: 'world' }, [], false).ok).toBe(false);
    expect(parseLibraryInput({ name: 'x', documents_language: 'klingon' }, [], false).ok).toBe(false);
    expect(parseLibraryInput({ name: 'x', documents_language: 'Unknown' }, [], false)).toEqual({
      ok: true, value: { name: 'x', visibility: 'private', documents_language: 'unknown' },
    });
    expect(parseLibraryInput({ documents_language: 'french' }, [], true).ok).toBe(false);
    expect(parseLibraryInput({ share_history: true }, [], true)).toEqual({ ok: true, value: { share_history: true } });
  });
});

describe('parseAskInput', () => {
  it('defaults model/k/focus and validates ids', () => {
    const r = parseAskInput({ question: ' hi ', k: 50, focus: { section: 'all' } });
    expect(r).toEqual({
      ok: true,
      value: { question: 'hi', model: 'sonnet', k: 5, focus: { documents: [], section: undefined },
        route: 'auto', library_ids: [], conversation_id: undefined },
    });
    expect(parseAskInput({ question: 'q', library_ids: ['nope'] }).ok).toBe(false);
    expect(parseAskInput({ question: 'q', library_ids: 'x' }).ok).toBe(false);
    expect(parseAskInput({ question: '' }).ok).toBe(false);
  });

  it('dedupes and caps library_ids, and defaults unknown routes to auto', () => {
    const id = (n: number) => `00000000-0000-0000-0000-${String(n).padStart(12, '0')}`;
    const r = parseAskInput({ question: 'q', library_ids: [id(1), id(1), id(2)], route: 'genie' });
    expect(r.ok && [r.value.library_ids, r.value.route]).toEqual([[id(1), id(2)], 'genie']);
    const bad = parseAskInput({ question: 'q', route: 'sql' });
    expect(bad.ok && bad.value.route).toBe('auto');
    expect(parseAskInput({ question: 'q', library_ids: Array.from({ length: 9 }, (_, i) => id(i)) }).ok).toBe(false);
  });
});

describe('parseFeedback', () => {
  it('accepts MyGPT ratings only', () => {
    expect(parseFeedback({ rating: 1, comment: ' ok ' })).toEqual({ ok: true, value: { rating: 1, comment: 'ok' } });
    expect(parseFeedback({ rating: 2 }).ok).toBe(false);
  });
});

describe('uploads', () => {
  it('allows ai_parse_document formats only', () => {
    expect(uploadExtension('Paper.PDF')).toBe('.pdf');
    expect(uploadExtension('notes.docx')).toBe('.docx');
    expect(uploadExtension('data.xlsx')).toBeNull();
    expect(uploadExtension('noext')).toBeNull();
  });

  it('sanitizes file names', () => {
    expect(sanitizeFilename('../../etc/passwd')).toBe('passwd');
    expect(sanitizeFilename('My Paper (v2).pdf')).toBe('My_Paper_v2_.pdf');
    expect(sanitizeFilename('résumé.pdf')).toBe('resume.pdf');
    expect(sanitizeFilename('...')).toBe('document');
  });
});
