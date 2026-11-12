# QiSens phase descriptions for zero shot phase scoring (written 2026-11-12, before any QiSens run)

**Status.** Frozen. This file is committed before any QiSens run of siglip_scores or vlm_scores and
is never edited after a QiSens result. A change would need a new file and a new rule.

**Sources.**
- **Phase names and order:** code/configs/steps.toml (five ordered process phases, each 1 to 4 minutes).
- **No phase definitions in the repo:** the FR document with the phase definitions is not in the
  repository (docs/ holds only the HLD and the LLD, and neither defines the phases). So every
  description below comes from the phase names and from general railway vehicle maintenance
  knowledge (bogies, wheelsets, axle boxes, brake parts, fasteners, lifting equipment, cleaning
  and measuring tools).
- **No QiSens data:** no QiSens frame, video or annotation was used.

**Format** (read by code/assemblyvision/analysis/phase_text.py):
- one `## ` heading per phase, in task order;
- a `Distinguishes:` line;
- three `> ` lines (the short version for the VLM prompt);
- `- ` lines, one visual description each (used as the SigLIP prompt ensemble).

Wide lines describe a fixed CCTV view of the workshop, close lines a handheld view of the hands.

## Removal

Distinguishes: the whole component is still being taken off the vehicle or bogie and moved away as one unit (against Disassembly, where an already removed unit is taken apart into its own parts at a bench).

> The worker detaches a whole component from the railway vehicle or bogie and takes it away.
> Typical: loosening the mounting bolts with a wrench or impact tool, attaching a hoist or crane hook, lifting the unit off, carrying or rolling it away.
> The part is still mounted or just coming off the vehicle; it stays in one piece.

- a worker loosening large mounting bolts on a train bogie with a long wrench
- a worker using an impact wrench on bolts that hold a component to the bogie frame
- a crane hook or chain hoist attached to a heavy part above a railway bogie
- a heavy metal component being lifted off a bogie by an overhead crane
- a worker guiding a suspended part away from the bogie with both hands
- a worker kneeling beside the wheelset under the vehicle, reaching up to unbolt a part
- a worker carrying a removed brake unit away from the vehicle
- a part being lowered onto a trolley or pallet next to the bogie
- a component being rolled away from the train on a wheeled cart
- the worker standing next to the vehicle, pulling a part free from its mounting
- a wide CCTV view of a railway workshop with a worker removing a part from a bogie
- a wide view of a train bogie on a pit or stand, a worker at its side detaching a component
- a wide view of an overhead crane lifting a part away from a railway vehicle
- a wide view of a worker walking away from the bogie carrying a part
- a close view of hands turning a wrench on a large bolt fixed to a bogie
- a close view of a bolt head and a socket wrench, the bolt being loosened
- a close view of hands unhooking a cable or hose from a mounted component
- a close view of a split pin or locking wire being pulled out before unbolting
- a close view of a component starting to separate from its mounting face
- a mounting face on the bogie, now empty, with bolt holes visible after the part came off
- protective gloves holding a lifting strap wrapped around a metal part
- the worker using a pry bar to free a stuck part from the vehicle

## Disassembly

Distinguishes: a component that is already off the vehicle is taken apart into its pieces, usually on a bench or stand, and parts come off one by one (against Assembly, where parts go on; against Removal, where the unit leaves the vehicle whole).

> The worker takes an already removed component apart into its parts, usually on a workbench or stand.
> Typical: unscrewing bolts and nuts, pulling off covers, bearings, springs and pins, laying the loose parts out in a row.
> Parts come off and pile up; the unit gets smaller and opens up.

- a worker unscrewing bolts from a metal housing on a workbench
- a worker pulling a cover plate off an axle box on a bench
- a worker removing a bearing from its housing with a puller
- loose bolts, nuts and washers laid out in a tray on a workbench
- several separated metal parts arranged in a row on a bench
- a worker taking a spring out of a brake or suspension unit
- a worker knocking out a pin with a hammer and drift
- a worker using a socket wrench on a component clamped in a vice
- a component opened up on a stand, its inner parts visible
- the worker lifting a small part off a larger assembly and setting it aside
- a wide CCTV view of a workbench where a worker takes a component apart
- a wide view of a worker at a bench beside a stand holding a dismantled part
- a wide view of parts spread across a bench and a worker working on the remaining unit
- a wide view of a worker bending over a part on a stand, removing pieces from it
- a close view of hands unscrewing a nut from a threaded stud
- a close view of hands lifting a cover off a metal housing
- a close view of a bearing being pulled out of a housing
- a close view of hands removing a seal or gasket from a groove
- a close view of a screwdriver prying a retaining ring out
- a close view of loose small parts being placed into a box or tray
- a dismantled brake caliper with its pads and pistons lying beside it
- a gear or wheel hub being separated from its shaft

## Cleaning

Distinguishes: parts are cleaned, not taken apart or measured: dirt, grease and rust come off and the surfaces get clean (against Disassembly before it, where parts come off, and Inspection after it, where parts are measured).

> The worker cleans parts and surfaces: dirt, old grease and rust are removed.
> Typical: brushing, wiping with cloths or rags, spraying cleaner or solvent, blowing with compressed air, washing in a parts washer.
> The parts change from dirty and greasy to clean and bright; no tools for measuring are used.

- a worker scrubbing a greasy metal part with a wire brush
- a worker wiping a metal surface with a cloth or rag
- a worker spraying cleaning fluid onto a dirty part
- a worker blowing a part clean with a compressed air gun
- a part being washed in a parts washer or cleaning tank
- a worker brushing dirt off a bearing housing
- dark grease and dirt being wiped off a shaft
- a worker using a scraper to remove old grease and deposits
- a bucket, brushes and rags next to a part on a bench
- a worker holding a spray bottle over a metal component
- a wide CCTV view of a worker cleaning parts at a washing station
- a wide view of a worker at a bench repeatedly wiping a part
- a wide view of a worker with a high pressure washer cleaning a large part
- a wide view of a worker moving between several parts with a brush and cloth
- a close view of hands scrubbing a metal surface with a brush
- a close view of a rag wiping black grease from a metal part
- a close view of compressed air blowing dust out of bolt holes
- a close view of foam or cleaning fluid running down a metal surface
- a close view of a clean shiny metal surface after wiping
- a close view of a dirty part beside an already cleaned part
- a worker cleaning the threads of a bolt with a brush
- a worker drying a cleaned part with a cloth

## Adjustment and Inspection

Distinguishes: parts are checked and measured, and settings are adjusted; hands hold gauges, calipers and meters, and nothing is taken apart or put together (against Cleaning before it and Assembly after it).

> The worker inspects and measures the cleaned parts and adjusts settings.
> Typical: measuring with a caliper, micrometer or gauge, looking closely at surfaces for cracks or wear, checking clearances, setting a value with a small tool, writing down results.
> The parts lie still while the worker looks, measures and notes; small careful movements.

- a worker measuring a part with a vernier caliper
- a worker using a micrometer on a shaft or bearing seat
- a worker checking a gap with a feeler gauge
- a worker holding a dial gauge against a rotating part
- a worker looking closely at a metal surface, checking it for cracks
- a worker shining a torch onto a part to inspect it
- a worker comparing a part against a gauge or template
- a worker writing measurement results on a clipboard or tablet
- a worker turning an adjustment screw by a small amount
- a wheel profile gauge held against a train wheel
- a wide CCTV view of a worker standing still at a bench, measuring a part
- a wide view of a worker bent over a part, looking at it closely
- a wide view of a worker with a clipboard next to a set of cleaned parts
- a wide view of a worker checking a mounted wheelset with a measuring tool
- a close view of hands holding a caliper across a metal part
- a close view of a micrometer scale being read
- a close view of a feeler gauge slipped into a narrow gap
- a close view of a dial indicator needle
- a close view of a fingertip tracing a surface, checking for wear
- a close view of a pen writing values on an inspection sheet
- a close view of a small screwdriver adjusting a setting screw
- a torque setting being dialled in on a torque wrench before use

## Assembly and Installation

Distinguishes: parts go on and are fastened, then the unit is mounted back onto the vehicle; the unit grows and bolts are tightened, often with a torque wrench (against Disassembly, where parts come off, and Removal, where the unit leaves the vehicle).

> The worker puts the parts back together and mounts the unit back onto the vehicle or bogie.
> Typical: fitting bearings, covers and pins, inserting and tightening bolts with a torque wrench, applying grease, lifting the unit into place with a hoist and bolting it on.
> Parts go on and get fixed; at the end the component is attached to the vehicle again.

- a worker fitting a cover plate onto a housing and inserting the bolts
- a worker tightening bolts with a torque wrench
- a worker pressing a bearing into its housing
- a worker applying fresh grease to a bearing or shaft
- a worker threading a nut onto a stud by hand
- a worker inserting a pin and fitting a split pin to lock it
- an assembled component on a stand with all its bolts in place
- a worker lifting a reassembled part into position on the bogie with a hoist
- a worker aligning a component with the bolt holes on the bogie frame
- a worker tightening the mounting bolts of a component on the vehicle
- a wide CCTV view of a worker mounting a part back onto a railway bogie
- a wide view of an overhead crane lowering a part into place on a bogie
- a wide view of a worker at a bench putting parts together
- a wide view of a worker tightening bolts around a component on the vehicle
- a close view of hands holding a torque wrench on a bolt until it clicks
- a close view of a bolt being screwed into a threaded hole
- a close view of hands pushing a part onto a shaft
- a close view of a seal being fitted into a groove
- a close view of grease being spread with a brush onto a metal surface
- a close view of a locking wire being twisted through bolt heads
- a close view of a component sitting flush on its mounting face with all bolts fitted
- a close view of paint marks being put on tightened bolts
