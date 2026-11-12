# IKEA ASM macro phase descriptions for zero shot phase scoring (written 2026-11-12, before any run)

**Status.** Frozen before any siglip_scores or vlm_scores run on IKEA ASM. It is never edited
after a result.

**Sources.** The macro phases of experiments/datasets/ikea_asm_macro_phases.md, the IKEA ASM
action names, and general knowledge of these IKEA products:
- Lack tables: a square table top, four screw in legs and, for the TV bench and the coffee table,
  a lower shelf;
- Kallax drawer: a front panel, two side panels, a bottom panel, a back panel and wooden pins.

No IKEA ASM frame was looked at to write them.

**Format** (the same as docs_team/phase_descriptions.md, read by analysis/phase_text.py):
- `# Furniture: <type>` sections, then one `## ` heading per phase in task order;
- a `Distinguishes:` line;
- three `> ` lines (short prompt version);
- `- ` lines (descriptions).

The videos are the top view camera, which looks down at the person and the furniture on the
floor or a table.

# Furniture: Lack

Shared by Lack_TV_Bench, Lack_Coffee_Table and Lack_Side_Table (the side table has no shelf
phase).

## Attach legs

Distinguishes: the table top lies upside down and legs are screwed into its corners one by one (against Flip table, where the whole table is turned over, and Attach shelf, where a flat shelf is added).

> The person screws the table legs into the corners of the table top, which lies upside down.
> Typical: picking up a leg, aligning its screw with the hole in a corner, spinning the leg round and round, tightening it.
> The top is upside down; one to four legs stick up from its corners.

- a person screwing a table leg into the corner of an upside down table top
- a person turning a square wooden leg round and round with both hands
- a table top lying upside down on the floor with one leg attached
- a table top lying upside down with two legs standing up from its corners
- an upside down table top with three legs attached and one leg on the floor
- a person picking up a table leg from the floor
- a person holding a leg upright over a corner of the table top
- a person aligning the screw of a leg with a hole in the table top
- a person kneeling beside an upside down table top, attaching a leg
- a person rotating the table top on the floor to reach the next corner
- a person flipping the flat table top over before attaching legs
- a view from above of a person screwing a leg into a table top
- a view from above of a square table top with legs standing up from it
- a view from above of a person bending over a table top on the floor
- spare table legs lying on the floor next to the table top
- hands twisting a wooden leg into a threaded corner
- a person tightening a leg with a final turn
- a person reaching across the table top to the far corner
- a person sitting on the floor next to the table top holding a leg
- a table top with no legs yet lying on the floor, a person picking up the first leg
- four legs pointing up from an upside down table top

## Flip table

Distinguishes: the table with its legs is turned over so it stands on its legs; a short whole body movement, no screwing (against Attach legs before it and Attach shelf after it).

> The person turns the table over so that it stands upright on its legs.
> Typical: grabbing the table by its edges or legs, lifting, rotating it in the air, setting it down on its legs, pushing it into place.
> The table goes from upside down to standing upright.

- a person lifting an upside down table with legs and turning it over
- a table in mid air being flipped over by a person
- a person setting a table down onto its four legs
- a table now standing upright on its legs
- a person holding a table by two legs and rotating it
- a person standing up while lifting the table
- a person pushing a standing table across the floor
- a view from above of a table being turned over
- a view from above of a table standing on its legs next to a person
- a person grabbing the edge of the table top with both hands
- a table on its side during the flip
- a person stepping back from the upright table
- the flat top of an upright table seen from above
- a person turning a table over with its legs pointing sideways
- a standing table being moved into position
- a person lowering the table carefully to the floor
- a person bending down to grab the table legs
- the table legs swinging through the air as the table turns
- a person checking that the table stands steady
- an upright small table and a person standing beside it
- a person with both arms raised holding the table above the floor

## Attach shelf

Distinguishes: a flat shelf board is fitted between the legs under the table top (against Attach legs, where legs are screwed in, and Flip table, where the table is turned).

> The person fits the lower shelf between the legs of the table.
> Typical: picking up the flat shelf board, turning it, sliding it between the legs, pressing it into place.
> A second flat board appears below the table top.

- a person picking up a flat shelf board from the floor
- a person sliding a shelf between the legs of a table
- a person holding a flat board under a table top
- a table with a lower shelf fitted between its legs
- a person pressing a shelf into place under the table
- a person flipping the shelf board over before fitting it
- a person laying the shelf board down on the floor
- a view from above of a person reaching under a table with a board
- a view from above of a table and a flat shelf board beside it
- a person kneeling beside the table, fitting the shelf
- a shelf board being angled between two legs
- a person aligning the shelf with the legs
- a table with two flat surfaces, the top and the shelf below
- hands pushing a board against the inside of a table leg
- a person adjusting the shelf so it sits level
- a flat white board under a black table top
- a person lifting the table slightly to fit the shelf
- a person checking the finished table with its shelf
- a long thin TV bench with a shelf underneath
- a coffee table with a magazine shelf being fitted
- the person standing next to a finished table with a shelf

# Furniture: Kallax_Shelf_Drawer

## Side panels

Distinguishes: the two side panels are pushed onto the dowels of the front panel, so an open frame forms (against Bottom panel, where a thin board slides in, and later phases).

> The person attaches the two side panels to the front panel of the drawer.
> Typical: picking up a side panel, aligning its holes with the wooden dowels of the front panel, pressing it on.
> An L or U shape of panels forms; there is no bottom yet.

- a person attaching a side panel to the front panel of a drawer
- a person aligning holes in a side panel with wooden dowels
- a person pressing a side panel onto the dowels of a front panel
- a front panel lying flat with wooden dowels sticking up
- an L shape of two panels, the front panel and one side panel
- a U shape of three panels, a front and two sides
- a person picking up a side panel from the floor
- a person holding a panel upright at a right angle to another panel
- a person laying down the front panel
- a view from above of two drawer panels joined at a corner
- a view from above of a person pressing panels together
- loose flat panels lying on the floor next to a person
- hands pushing a panel firmly onto dowels
- a person tapping a panel to seat it on the dowels
- a person holding the front panel steady with one hand
- small wooden dowels in the edge of a panel
- a person turning a panel around to find the holes
- a person kneeling over a set of drawer panels
- an open drawer frame without a bottom
- a person checking that the side panel is straight
- two side panels standing up from the front panel

## Bottom panel

Distinguishes: a thin flat bottom board is slid into the grooves of the frame (against Side panels, where thick panels are pressed onto dowels, and Back panel, which closes the frame).

> The person slides the thin bottom panel into the grooves of the drawer frame.
> Typical: picking up the thin board, inserting its edge into the grooves of the side panels, pushing it in.
> The open frame gets a flat bottom.

- a person sliding a thin board into the grooves of a drawer frame
- a person picking up a thin bottom panel
- a person pushing a bottom panel along the grooves
- a drawer frame with a thin board half slid in
- a drawer frame with a flat bottom in place
- a person holding a large thin board level
- a person laying down the bottom panel
- a view from above of a drawer frame with a bottom panel going in
- a view from above of a thin board next to an open frame
- hands guiding the edge of a thin board into a groove
- a person bending over the frame, pushing the board
- a thin white board between two side panels
- a person adjusting the bottom so it sits in both grooves
- a person kneeling in front of the drawer frame
- an almost complete drawer box without a back
- a person holding the frame steady while sliding the bottom
- a person lifting the bottom panel off the floor
- the bottom panel sliding under the front panel edge
- a person checking that the bottom has gone all the way in
- a person tilting the frame to slide the panel in
- a board disappearing into the frame grooves

## Back panel

Distinguishes: the back panel is attached to close the frame, so the drawer becomes a closed box (against Bottom panel before it and Pins and finish after it).

> The person attaches the back panel and closes the drawer box.
> Typical: picking up the back panel, aligning it with the ends of the side panels, pressing it on.
> The drawer becomes a closed rectangular box.

- a person attaching the back panel to a drawer box
- a person aligning a back panel with the ends of two side panels
- a person pressing the back panel into place
- a closed rectangular drawer box
- a person picking up the back panel from the floor
- a person laying down the back panel
- a view from above of a drawer box with four walls
- a view from above of a person closing a drawer frame with a panel
- hands pushing a panel onto the end of a frame
- a person holding the box while fitting the last panel
- a drawer box with its back panel standing loose beside it
- a person kneeling behind the drawer box
- a person checking the corners of the box
- the back panel covering the bottom panel edge
- a person tapping the back panel into place
- a person turning the box to reach the back
- four panels forming a closed square frame
- a person squaring the box with both hands
- a drawer that looks complete from above
- a person lifting the box slightly to fit the back
- the last open side of the drawer frame being closed

## Pins and finish

Distinguishes: small pins are pushed in to lock the box, and the drawer is turned right side up; small precise hand movements, then the finished drawer (against Back panel, where a panel is attached).

> The person inserts small pins to lock the drawer and turns the finished drawer right side up.
> Typical: picking up small pins, pushing them into holes in the panels, pressing them in, turning the drawer over.
> The drawer is complete; small parts go in, no panels.

- a person pushing a small pin into a hole in a drawer panel
- a person picking up a small pin from the floor
- a person pressing pins into the corners of a drawer
- fingertips holding a tiny wooden pin
- a finished drawer box being turned right side up
- a complete drawer standing the right way up
- a view from above of a person inserting pins into a box
- a view from above of a finished drawer next to a person
- a person turning the drawer over with both hands
- a person checking the drawer is finished
- small pins lying on the floor next to the drawer
- a person bending close to the box, fitting small parts
- hands pressing a pin flush with a panel
- a person going around the box inserting pins
- a person setting the finished drawer down
- a closed drawer with no loose panels around it
- a person standing up next to the finished drawer
- a person sliding the drawer to check it
- a person pressing on a corner to lock it
- a person picking up the last pin
- a drawer that is complete and right side up
