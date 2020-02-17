* utils > count_avg_gradient:
  * `lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)    # different if not convert from uint8`
  * originally `lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)` without cast to float32 is uint8 would cause digit overflow during running kernal.