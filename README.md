Running verifiers on Modal
=====
- This was written before @willccbb pushed a bunch of new updates to `verifiers`, as such it uses the old version of that code
- At the time, `verifiers` would not import/run on recent versions of Python. I had to fork `verifiers` and patch a few things to get it running. As such, the image in this Modal App builds from [my fork of verifiers](https://github.com/jvmncs/verifiers) @ `main`.
- If I was doing this for real, I'd make this function fault-tolerant/resumeable (so would adapt `math_train.py` to load from checkpoints/resume properly).

In general, I'd advise to update to the new version of verifiers and build the image from that. I will do that to this code whenever I get the chance (or at some point might move this example in that updated form elsewhere).
