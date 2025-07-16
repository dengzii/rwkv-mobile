#include <chrono>
#include <iostream>

#include "runtime.h"


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    rwkvmobile::runtime runtime;

    const auto model_path = std::string(argv[1]);
    runtime.init_embedding(model_path);

    const std::string text1 =
            "Immanuel Kant[a] (born Emanuel Kant; 22 April 1724 – 12 February 1804) was a German philosopher and one of the central thinkers of the Enlightenment."
            "Born in Königsberg, Kant's comprehensive and systematic works in epistemology, metaphysics, ethics, and aesthetics have made him one of the most influential and highly discussed figures in modern Western philosophy."
            "In his doctrine of transcendental idealism, Kant argued that space and time are mere 'forms of intuition that structure all experience and that the objects of experience are mere 'appearances."
            "The nature of things as they are in themselves is unknowable to us. "
            "Nonetheless, in an attempt to counter the philosophical doctrine of skepticism, he wrote the Critique of Pure Reason (1781/1787), his best-known work. "
            "Kant drew a parallel to the Copernican Revolution in his proposal to think of the objects of experience as conforming to people's spatial and temporal forms of intuition and the categories of their understanding so that they have a priori cognition of those objects."
            "Kant believed that reason is the source of morality and that aesthetics arises from a faculty of disinterested judgment. "
            "Kant's religious views were deeply connected to his moral theory. "
            "Their exact nature remains in dispute."
            "He hoped that perpetual peace could be secured through an international federation of republican states and international cooperation. "
            "His cosmopolitan reputation is called into question by his promulgation of scientific racism for much of his career, although he altered his views on the subject in the last decade of his life.";

    const auto &t = text1;

    std::vector<std::string> seg;
    size_t start = 0;
    size_t end = t.find('.');

    while (end != std::string::npos) {
        seg.push_back(t.substr(start, end - start));
        start = end + 1;
        end = t.find(',', start);
    }
    seg.push_back(t.substr(start));

    std::cout << "seg size = " << seg.size() << std::endl;

    auto now = std::chrono::high_resolution_clock::now();
    for (const auto &part: seg) {
        const auto embd = runtime.embed(part);
        std::cout << '.';
    }
    std::cout << std::endl;
    auto elapsed = std::chrono::high_resolution_clock::now() - now;
    std::cout << "elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms" <<
            std::endl;

    return 0;
}
