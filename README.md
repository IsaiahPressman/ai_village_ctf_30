# ai_village_ctf_30

This repository contains my solutions for the [AI Village Capture the Flag @ DEFCON](https://www.kaggle.com/competitions/ai-village-ctf) Kaggle competition.
I won the competition by being the first competitor to complete 21 out of the 22 problems. No one solved the 22nd problem. Here is a brief write-up of my experience, copied from the [Kaggle forums](https://www.kaggle.com/competitions/ai-village-ctf/discussion/353536):

First of all, thanks to the organizers for putting on a wonderful competition! I had a ton of fun with this novel format, and found the fast puzzle-solving nature of it quite addicting. The weekend when this competition launched was a long and sleep-deprived one! Many other competitors have gone over their solutions, and I don't have much to add, but I will briefly cover my general strategic approach and the solutions I found for each of the problems. A particular shoutout to [Chris Deotte's notebook](https://www.kaggle.com/code/cdeotte/solutions-d3fc0n-ctf-lb-0-894), which is well written and solves all of the problems in a principled way. For those looking for the code to my solutions, it is rather messy, but can be found on GitHub: [https://github.com/IsaiahPressman/ai_village_ctf_30](https://github.com/IsaiahPressman/ai_village_ctf_30)

Generally, my strategy for this challenge was to delegate as much work as possible to an inelegant and/or brute-force solution in order to free up my time to work on the next problem. While this was not as intellectually satisfying as finding the proper method, it did allow me to move through the problems quite quickly. My solutions, in the order I solved them, are as follows:

- Hotdog: Submitted a picture of a hotdog.
- Math_1-4: Brute force starting at 100 and incrementing until the solution was found.
- Honor Student: Used an image editor to draw an A, and then an online image compressor to avoid tampering detection.
- Wifi: Visualized the data projected into 2 dimensions using a few different locally linear embedding algorithms from scikit-learn. Picked the one that looked the cleanest and projected the characters into 1 dimension, to find the flag.
- Bad to Good: Manually tinkered with the inputs until it worked. Negative demerits ended up being the trick.
- Baseball: Grid searched over a bunch of values to be the mean of two independent normal distributions, and then ordered by their returned confidence. Then just tried a bunch of normally distributed values around the highest confidence mean values from the previous step until one worked.
- Inference: Handwrote every letter and digit and converted them into 32x32 grayscale images. Submitted these images to the server to see which character the model thought belonged where, and then brute force searched over the top 5 candidate characters for each position.
- Leakage: Fed the username into the LSTM and it returned the password.
- Forensics: Found the flag using model.summary()
- Token: Opened the data in excel and saw a bunch of lines at the end with BLANK and SECRETKEY. Searched for any other occurrences of those words in the file, and found two lines with BLANK occurring twice, which were the solution.
- Deepfake: Edited the video by replacing the video track with a still image of the boss.
- Murderbots: Trained a basic logistic regression model, and submitted the 10 indices that the model ranked as most likely to be human.
- Hotterdog/Theft/Salt: All of these required adversarial examples. Some of them (Salt/Theft) were easier since they gave you the model, while with Hotterdog I brute-force searched across a bunch of open-source models + adversarial attacks, and combined those results with a noisy overlay of an actual hotdog image.
- Crop_1: Tricky problem, until I realized I could submit an image of a resolution other than 5x5. After that, I used a 3x3 grid of images that more or less matched X_comp so that it didn't matter where the model cropped.
- WAF: This problem or sloth was the most exciting to finally solve. The trick for me was to reverse engineer the malicious string by submitting a fixed 4-character chunk and varying the final character until the WAF detector was triggered, and then repeat using the final 4 characters of the new 5-character malicious chunk. (And do the same thing in reverse to find the beginning of the string) Once I had the full exploit, I decoded it and added some spaces + a variable before re-encoding to bypass the WAF.
- Secret Sloth: This problem took me a while, but eventually I had the insight that I could brute-force search over a dictionary in the hopes that the solution would be a single word. It was.
- Crop_2: :(

All in all, this was a great competition! The challenges were engaging, and I'm grateful to have had the time available early on in the competition to solve so many of them. One thing I was amused by was the Easter eggs/"true flags" of sorts hidden in the flags. Each challenge's genChallengeFlag() method had a set of characters that remained fixed from flag to flag, and were usually an anagram of something related to the challenge. I wasn't able to figure all of them out, but for those interested, here's what I have so far:

| Challenge ID | Proto-flag characters | My proto-flag hypothesis |
| --- | --- | --- |
| math_1 | 8T3I | ? |
| math_2 | EITMACHATMA | ? |
| math_3 | HOMR_IAPTTM | ? |
| math_4 | EWPTIIRNMS | ? |
| honorstudent | SOOPATESHCFTL | ? |
| bad_to_good | wtphn_t_eabo_d | dont_be_a_wtph ? |
| baseball | ERAJAOMGLEU | ? |
| token | 3RTK330RN1YM0DZB3 | ? |
| waf | RT0ZB0DLILEYN0RKAZY | ? |
| theft | RTCH1P?0N1LYNDD33P | ? |
| salt | 334TTL<RSLBGS14LGL1TT | ? |
| crop1 | OnnTeghraTag | ? |
| hotdog | OOSHDRSYFOATDG | HOTDOGFORDAYSS |
| hotterdog | OOTHDDDTHOHGGGOOOT | HOTDOGHOTDOGHOTDOG |
| deepfake | ? (server kept failing) | ? |
| forensics | 0X53NRD1FC_S | F0R3NS1C5_XD |
| wifi | NERDTU | TURNED |
| leakage | as4rnoH<c33<0RN | s4RaHc0Nnor<3<3 |
| sloth | PCESALTR | SPECTRAL |
| murderbots | beotautrsdtodWIeNM | IWasNotMurdebotted |
| inference | 30NDCF | D3FC0N |
| crop_2 | ? | ? |

