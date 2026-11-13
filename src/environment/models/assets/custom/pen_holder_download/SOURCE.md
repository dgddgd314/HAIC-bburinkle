# Pen Holder source and license

This work is based on [Pen Holder](https://sketchfab.com/3d-models/pen-holder-a29fc7fea63c4689bf2b9b6f7b50eb44)
by [salsikash101](https://sketchfab.com/salsikash101), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The conversion input is the user-provided
`molmospaces/pen_holder_obj2/pen_holder.obj`. Its referenced texture files were
not supplied, so the source uses an opaque graphite-coated metal material.
The visual uses the actual obj2 mesh. Collision uses a solid bottom and 48
overlapping wall segments fitted to its measured opening, so the functional
top cavity stays open while the side wall has no collision gaps.
The outer diameter is set to 16.0 cm and height to 16.07 cm. The wall
thickness remains about 9.58 mm rather than growing with the diameter, giving
a usable inner diameter of about 14.08 cm. The radial profile and height ratio
remain close to the supplied asset, and collision uses the same dimensions.
