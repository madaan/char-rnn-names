from collections import defaultdict
import numpy as np
from string import ascii_lowercase


class CharCodec:
    """
    Wraps all the information required to encode characters in the names to integers and vice versa.
    The encoder understands the small case english alphabet (mapped 0-25), space (mapped to 26) and the end of the name
    character # (mapped to 27). Everything else is mapped to 28.
    """

    max_name_length = 25 + 1 #(+1 to compensate for the NAME_END)
    INVALID_CHAR_CLASS = 28
    char_to_class = defaultdict(lambda: CharCodec.INVALID_CHAR_CLASS)

    for c in ascii_lowercase:
        char_to_class[c] = ord(c) - ord("a")
    char_to_class[" "] = 26 #space

    #end of the name, added to every name. This character is important for the model since it implicitly models the
    #typical name lengths
    NAME_END = "#"
    char_to_class[NAME_END] = 27

    #characters that are not understood
    INVALID_CHAR = "."
    char_to_class[INVALID_CHAR] = INVALID_CHAR_CLASS

    class_to_char = {v:k for k, v in char_to_class.items()}

    @staticmethod
    def encode(name):
        """
        :param name: A string, like "abcd"
        :return: maps each character in the string to the corresponding class and returns a list of classes.
        For example, "abcd" -> [0, 1, 2, 3]

        """
        return [CharCodec.char_to_class[c] for c in name.lower()]

    @staticmethod
    def decode(classes):
        """

        :param classes: a list of integers such that each "c" in classes is between [0, 28]
        :return: The decoded characters, one per class.
        """
        return [CharCodec.class_to_char[c] for c in classes]


    @staticmethod
    def encode_and_standardize(name):
        """
        :param name: A string name, like "rishab arora".

        :return: The name with each character encoded to the corresponding class, and padded/truncated to
        max_name_length.
        The name end symbol is added to the end of every name. All the characters after the end of the name are mapped
        to INVALID_SYMBOL.
        The following set of operations are applied to the given name:
        ankur dewan (length 11, max_name_length 25) ->
        1. Add end of name
            ankur dewan# (length 12)
        2. Add 13 invalid symbols to make the length 25
            ankur dewan#.............
        3. Return encoded version of the string "ankur dewan#............." using CharacterEncoding.char_to_class dict.
        [ 0 13 10 20 17 26  3  4 22  0 13 27 28 28 28 28 28 28 28 28 28 28 28 28 28]
        The mapping is performed first in the implementation, but the order doesn't matter.

        For names longer than or equal to the max_name_length, the NAME_END character is added in the last position
        (names[max_name_length - 1], and rest of the name is truncated.)
        """
        name = str(name)
        name = name + CharCodec.NAME_END #add the end of the name symbol for everyname
        name = CharCodec.encode(name) #encode
        name_len = len(name)

        if name_len >= CharCodec.max_name_length:
            truncated_name = name[:(CharCodec.max_name_length - 1)]
            truncated_name.append(CharCodec.char_to_class[CharCodec.NAME_END]) #must attach the name end
            return np.array(truncated_name, dtype=np.int32)
        else:
            padded_name = np.empty(CharCodec.max_name_length, dtype=np.int32)
            padded_name.fill(CharCodec.INVALID_CHAR_CLASS)
            padded_name[:name_len] = name
            return padded_name

if __name__ == "__main__":
    print(CharCodec.encode_and_standardize("ankur dewan"))
    print(CharCodec.encode_and_standardize("really really really really really really long random name"))
    print(CharCodec.encode_and_standardize(""))
    print(CharCodec.encode_and_standardize("p"))