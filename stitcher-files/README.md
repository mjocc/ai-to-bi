index.tsx, results.tsx replace their corresponding files in the capture part of the main app
useBeeStore.ts replaces the existing store file 

stitcherApi.ts goes into the services part of the main app

stitcher.py and stitcher_api.py should be in the same folder (beehive_stitcher) with an \__init__.py file (can be blank)
There should be another folder named "panoramas" on the same level as this folder
stitcher_api uses uvicorn to host, instructions are in there (needs a terminal of its own)
