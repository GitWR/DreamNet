1. This is the source codes of our proposed DreamNet that has been accepted by Asian Conference on Computer Vision (ACCV) 2022.

2. These three subfolders that named DreamNet-27, DreamNet-47, and DreamNet-92 respectively correspond to the implementations of the 27/47/92-layer DreamNets studied in our article.

3. Since our implementation is based on the source code of SPDNet, we would like to express our sincere thanks to the published official code of [1].
[1] Huang, Zhiwu, and Luc Van Gool. "A Riemannian Network for SPD Matrix Learning." Thirty-First AAAI Conference on Artificial Intelligence, 2017.

4. To run our network, the Matlab 2019 (or higher version) software with the deep learning toolbox is required. You can select this toolbox and other necessary 
computing components according to the system prompts during the installation of this software. After completing the above step, you need to put the files named
FPHA (we here take the FPHA dataset as an example) and SPD_info.mat under the path '.\data\afew\'.
Then, run spdnet_afew.m to obtain the classification performance versus the number of training epochs.