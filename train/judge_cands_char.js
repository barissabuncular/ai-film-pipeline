export const meta = {
  name: 'character-candidate-judge',
  description: 'Judge LoRA training candidates for one character: same person, kit, style, requested view/place/light, crop box',
  phases: [ { title: 'Judge', detail: 'one judge per group of folders' } ],
}
// args = { dir, meta, groups: [[folder,...],...], who: 'Moreina, a rogue archer', refs: ['<project root>/train/pilot/<char>/A0.png', ...],
//          checklist: 'face and kit, where visible', rules: 'extra keep rules' }
if (!args || !args.dir || !Array.isArray(args.groups) || !Array.isArray(args.refs)) throw new Error('args must be {dir, meta, groups, who, refs, checklist, rules}')
const J = { type: 'object', properties: {
  images: { type: 'array', items: { type: 'object', properties: {
    file: { type: 'string' }, keep: { type: 'string', enum: ['yes', 'maybe', 'no'] },
    same_person: { type: 'string', enum: ['yes', 'close', 'no', 'face not visible'] },
    kit_ok: { type: 'string', enum: ['yes', 'minor issue', 'no'] },
    style: { type: 'string', enum: ['photoreal film still', 'slightly glossy or CG', 'cartoon, game art or painting'] },
    place_ok: { type: 'string', enum: ['yes', 'partly', 'no', 'not asked'] }, light_ok: { type: 'string', enum: ['yes', 'partly', 'no', 'not asked'] },
    anchor_background: { type: 'boolean', description: 'the anchor background (e.g. a plain grey studio) appears where a different place was asked' },
    actual_view: { type: 'string', enum: ['front view', 'three-quarter view', 'side view', 'back three-quarter view', 'back view', 'low-angle view from the front'] },
    actual_shot: { type: 'string', enum: ['close-up', 'medium shot', 'full body'] },
    states: { type: 'string' }, flaws: { type: 'string' },
    waist_up_box: { type: 'string', description: 'full-body images only: "x0,y0,x1,y1" in image pixels for a clean waist-up crop (head inside, about square), else ""' } },
    required: ['file', 'keep', 'same_person', 'kit_ok', 'style', 'place_ok', 'light_ok', 'anchor_background', 'actual_view', 'actual_shot', 'states', 'flaws', 'waist_up_box'] } },
  summary: { type: 'string' } }, required: ['images', 'summary'] }
const res = await parallel(args.groups.map(g => () => agent(
  `You judge LoRA training candidates for one character (${args.who}) for a Diablo fan film. Consistency is the user's top priority.
References (Read first): ${args.refs.join(', ')}.
Identity and kit checklist, where visible: ${args.checklist}
The requested shot, view, action, place and light for each folder are in ${args.meta} (key = folder name; missing fields were not asked).
Open with Read EVERY png in these folders under ${args.dir}: ${g.join(', ')}.
"keep: yes" only if: the RENDERING STYLE is a photoreal live-action film still (any cartoon, game-art, painted or smooth glossy 3D look is "no"); the same person (face, hair,
age); the kit is complete for what is visible; the requested action, place and light are shown where requested; one copy of the character only; no extra limbs, no warped
hands or weapons; no text or modern objects. ${args.rules || ''}
"maybe" = fixable by a crop. Report the ACTUAL view and shot, and in states list visible variable things worth captioning (an action, an expression, a prop in use).
Paths relative to ${args.dir}.`,
  { label: `judge:${g[0]}`, phase: 'Judge', schema: J }).then(r => r && ({ groups: g, ...r }))))
return res.filter(Boolean)
