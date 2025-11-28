import cv2
import os

# ---------- LOAD LOGOS ----------
def load_logos(folder_path):
    logos = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            brand_name = filename.split(".")[0]
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_COLOR)

            if img is None:
                print(f"Error loading {filename}")
                continue

            logos[brand_name] = img

    return logos


# ---------- EXTRACT FEATURES ----------
def extract_features(logos):
    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.1, nlevels=12)
    logo_features = {}

    for brand, img in logos.items():
        keypoints, descriptors = orb.detectAndCompute(img, None)

        if descriptors is None:
            print(f"⚠ No features found for {brand} — change the logo image!")
            continue

        logo_features[brand] = {
            "image": img,
            "keypoints": keypoints,
            "descriptors": descriptors
        }

        print(f"Extracted {len(keypoints)} features for {brand}")

    return logo_features


# ---------- MATCHING (Improved) ----------
def match_logo(frame_des, logo_des):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(logo_des, frame_des, k=2)

    good = []

    for m, n in matches:
        # Strong ratio test + distance threshold
        if m.distance < 0.70 * n.distance and m.distance < 40:
            good.append(m)

    return good



# ---------- MAIN DETECTION ----------
def detect_logos_webcam(logo_features):
    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.1, nlevels=12)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam!")
        return

    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Can't read webcam frame")
            break

        frame_key, frame_des = orb.detectAndCompute(frame, None)
        detected_brand = None
        best_score = 0
        best_confidence = 0

        if frame_des is not None:
            for brand, data in logo_features.items():
                matches = match_logo(frame_des, data["descriptors"])
                good_matches = len(matches)

                confidence = int((good_matches / 50) * 100)
                if confidence > 100:
                    confidence = 100

                if good_matches > best_score and good_matches >= 30:
                    best_score = good_matches
                    detected_brand = brand
                    best_confidence = confidence

        # Only show result if strong enough
        if detected_brand and best_confidence >= 40:
            cv2.putText(frame, f"Brand: {detected_brand} ({best_confidence}%)",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3)

        cv2.imshow("Brand Logo Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# ---------- RUN ----------
if __name__ == "__main__":
    folder = "../logos"

    logos = load_logos(folder)
    features = extract_features(logos)

    print("\nStarting webcam detection...")
    detect_logos_webcam(features)
