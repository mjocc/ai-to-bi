/**
 * src/services/stitcherApi.ts
 *
 * Set STITCHER_BASE_URL to your machine's LAN IP, e.g. "http://192.168.1.42:8000"
 */

const STITCHER_BASE_URL = "http://192.168.0.175:8000"; // update this

export async function stitchVideo(videoUri: string): Promise<string> {
  console.log("Sending video URI:", videoUri);

  const formData = new FormData();
  formData.append("video", {
    uri: videoUri,
    name: "hive_recording.mp4",
    type: "video/mp4",
  } as any);

  let response: Response;
  try {
    response = await fetch(`${STITCHER_BASE_URL}/stitch`, {
      method: "POST",
      body: formData,
    });
  } catch (e: any) {
    throw new Error(
      `Could not reach stitcher at ${STITCHER_BASE_URL} - is the server running and on the same WiFi? (${e?.message})`
    );
  }

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Stitcher API error ${response.status}: ${body}`);
  }

  const data: { panorama_url: string } = await response.json();

  if (!data.panorama_url) {
    throw new Error("Stitcher response missing panorama_url");
  }

  return `${STITCHER_BASE_URL}${data.panorama_url}`;
}