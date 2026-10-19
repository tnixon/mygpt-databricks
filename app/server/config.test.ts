import { describe, expect, it } from 'vitest';
import { volumePath, volumeSchema } from './config';

describe('volume config', () => {
  it('accepts the injected /Volumes path or a dotted name', () => {
    expect(volumePath('/Volumes/cat/sch/uploads')).toBe('/Volumes/cat/sch/uploads');
    expect(volumePath('cat.sch.uploads')).toBe('/Volumes/cat/sch/uploads');
  });

  it('derives the governed catalog.schema from the uploads volume', () => {
    expect(volumeSchema('/Volumes/cat/sch/uploads')).toBe('cat.sch');
    expect(volumeSchema('cat.sch.uploads')).toBe('cat.sch');
  });
});
