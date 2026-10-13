export const meta = {
  name: 'character-eval-judge',
  description: 'Blind-judge a character LoRA evaluation grid (neutral codes X01..), one judge per pair of scenes',
  phases: [ { title: 'Judge', detail: 'one judge per pair of scenes, all columns blind' } ],
}
// args = { blind: '<project root>/train/pilot/eval_<name>_blind', codes: ['X01', ...], scenes: ['e01_doorway', ...],
//          who: 'the Butcher, a demon', checklist: '<identity and kit, where visible>', notes: '<extra per-character rules>' }
// Save the result as train/pilot/<char>/eval_<name>_judge.json, then run train/readout_char.py.
if (!args || !args.blind || !Array.isArray(args.codes) || !args.codes.every(x => /^X\d\d$/.test(x)) || !Array.isArray(args.scenes))
  throw new Error('args must be {blind, codes, scenes, who, checklist, notes}')
const D = args.blind
const schemaFor = g => ({ type: 'object', properties: {
  rows: { type: 'array', items: { type: 'object', properties: {
    scene: { type: 'string', enum: g }, column: { type: 'string', enum: args.codes }, seed: { type: 'integer' },
    identity: { type: 'number' }, costume: { type: 'number' }, prompt_following: { type: 'number' }, background_bleed: { type: 'number' },
    artifacts: { type: 'number' }, style_photoreal: { type: 'number' },
    face_view: { type: 'string', enum: ['front', 'three-quarter', 'profile', 'back', 'hidden'] },
    extra_object: { type: 'boolean' }, notes: { type: 'string' } },
    required: ['scene', 'column', 'seed', 'identity', 'costume', 'prompt_following', 'background_bleed', 'artifacts', 'style_photoreal', 'face_view', 'extra_object', 'notes'] } },
  summary: { type: 'string' } }, required: ['rows', 'summary'] })
const groups = []
for (let i = 0; i < args.scenes.length; i += 2) groups.push(args.scenes.slice(i, i + 2))
const judged = await parallel(groups.map(g => () => agent(
  `You evaluate images from a character LoRA test (${args.who}) for a Diablo fan film. Open only files under ${D}; do not list or open any other folder.
Read the references first: every image in ${D}/_ref/.
The exact prompt of every image is in ${g.map(s => `${D}/prompts_${s}.json`).join(' and ')} (key "<scene>_<code>/kf_<seed>.png"). The codes are anonymous variants in shuffled order; judge only what you see, never guess what a code is.
For each scene in [${g.join(', ')}], open every image in ${D}/<scene>_<code>/ for the codes ${args.codes.join(', ')} (3 seeds each, kf_<seed>.png). Score each image 0-10:
- identity: the same character as the references. ${args.checklist}
- costume: the reference's kit, judged on the items this image's prompt names and the framing shows. Do not penalise an item the prompt leaves out.
- prompt_following: camera, pose, action, place, light and any stated state (for example blood) as written in this image's prompt.
- background_bleed: 10 = no training background repeating where not asked; 0 = a reference background replaces the asked one.
- artifacts: 10 = clean; deduct for blur, ghosting, extra limbs, warped hands or weapons, melted faces.
- style_photoreal: 10 = a photoreal live-action film still; 0 = cartoon, game art or a toy-like 3D render.
- face_view: front, three-quarter, profile, back or hidden. extra_object: a doubled weapon or shield, or a second copy of the character.
${args.notes || ''}
Score every image on its own; do not rank codes against each other. Be concrete in notes.
Return one row per image: scene = the scene name exactly as listed, column = the code only (for example X03), seed = the number in kf_<seed>.png.`,
  { label: `judge:${g[0]}`, phase: 'Judge', schema: schemaFor(g) })))
return { judged: judged.filter(Boolean), groups_missing: judged.filter(j => !j).length }
