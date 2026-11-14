# Attribution and redistribution review

Reviewed 2026-11-14. This local package has not been published.

- Application source: Copyright (c) 2026 Nicolás Florentín, MIT; full notice in LICENSE.
  Source: https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition
  at 45ff2c0e26925c50f4546073bd573fe3ca7a5e6e, plus the user's local completed changes.
- GestureSign preprocessing, adapted decision/segmentation behavior, fixtures,
  and assets: Copyright (c) 2026 Shashank Yadav. MIT notice retained verbatim in
  vendor/GestureSign/LICENSE and WORD_RECOGNITION.md. Source:
  https://github.com/shashank11yadav/GestureSign at
  ffc0578e68449676b6e9545da82092679914e8c8. Upstream README retained alongside it.
- MediaPipe hand styling is transcribed in preprocessing.py. MediaPipe is
  Copyright The MediaPipe Authors, Apache-2.0. Full license in docs/MEDIAPIPE_LICENSE;
  source https://github.com/google-ai-edge/mediapipe. The palette transcription
  is from the original project; this packaging does not modify it.

## Alphabet model: unresolved redistribution permission

The source LICENSE explicitly excludes asl_cnn_model.onnx weights from its MIT
permission. Its README identifies ASL Alphabet by Akash Nagaraj (GPL-2.0),
ASL Dataset by Ayush Thakur (CC0-1.0), and ASL Alphabet Test by Dan Rasband
(CC0-1.0). Dataset links and the original qualification about derivative
weights are preserved in docs/ORIGINAL_README.md. No separate affirmative
weight license was found in the source project. Do not treat the code's MIT
license as weight redistribution permission. Obtain clarification from the
model owner and assess the dataset terms before publishing these weights.

## Word model and replay clips

GestureSign ships model.onnx under a repository containing an MIT license;
its model card identifies Google Isolated Sign Language Recognition / PopSign
(asl-signs) data as CC BY 4.0. The six replay clips retain their full embedded
credit and provenance in vendor/GestureSign/web/public/replay/clips.json;
the replay CLI prints that credit. Model bytes, metadata, labels, and replay
content are unchanged. See https://github.com/shashank11yadav/GestureSign and
https://www.kaggle.com/competitions/asl-signs/data . CC BY 4.0 terms:
https://creativecommons.org/licenses/by/4.0/ . Retain attribution, license link,
and modification notices when redistributing these assets.

The local notices and upstream repository support that provenance, but an
explicit model-specific grant distinct from the repository MIT license was
not located. Kaggle's dataset pages did not expose readable terms during the
web check. Confirm the applicable asset/competition terms before publishing;
this review does not resolve the alphabet weights' missing license.
