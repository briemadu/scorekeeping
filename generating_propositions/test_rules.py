#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Unit tests for the generation rules.
"""

import unittest

from collections import Counter

from aux import generate_proposition, propositions_from_caption
from rules import (what_color, can_you_see, there, any_, weather, what_kind,
                   is_it, is_inside, in_answer, out_answer, sunny, cloudy,
                   weather_answer, daytime, daytime_answer, image_in_color,
                   person_answer, no_people, is_noun_adj, noun_,
                   noun_prep_noun, noun_verb, noun_verb_noun, have, look_adj,
                   noun_noun, what_is, adj_or_adj,
                   is_np_np, verb_pron_adj, verb_pron_obj, pron_prep,
                   look_like, removeprefix, removesuffix)


class TestManipulationRules(unittest.TestCase):

    def test_generate_proposition(self):
        """Check whether the right rule is being captured for types of qa."""

        q, a = ("what color is the car", "i don\'t know")
        result = generate_proposition(q, a, q, a)
        self.assertEqual(result, [])

        q, a = ("what color are the sheep", "dirty white")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('what_color', rule)

        q, a = ("what color walls", "white")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('what_color', rule)

        q, a = ("can you see a sink", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('can_you_see', rule)

        q, a = ("are there any animals in the image", "no")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('there', rule)

        q, a = ("any lakes nearby", "no")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('any_', rule)

        q, a = ("what kind of food", "asian")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('what_kind', rule)

        q, a = q, a = ("is it sunny", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('sunny', rule)

        q, a = ("is it cloudy", "no")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('cloudy', rule)

        q, a = ("is it daytime", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('daytime', rule)

        q, a = ("is it a boy", "yes, it is")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('is_it', rule)

        q, a = ("what is the weather like", "the weather is beautiful")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('weather', rule)

        q, a = ("is this outdoors", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('is_inside', rule)

        q, a = ("can you tell if the image is outside", "inside")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('in_answer', rule)

        q, a = ("can you tell if the image is indoors", "outdoors")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('out_answer', rule)

        q, a = ("is the photo in color", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('image_in_color', rule)

        q, a = ("do you know what time is it", "daytime")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('daytime_answer', rule)

        q, a = ("is the weather sunny", "overcast")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('sunny/cloudy', rule)

        q, a = ("can you tell whether it is a male", "female")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('male/female', rule)

        q, a = ("what are the people doing", "no people")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('no_people', rule)

        q, a = ("is sky visible", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('is_noun_adj', rule)

        q, a = ("gloves", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('noun_', rule)

        q, a = ("are the owners in the picture", "no")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('noun_prep_noun', rule)

        q, a = ("are the vegetables cooked", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('noun_verb', rule)

        q, a = ("is the man wearing a hat", "no")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('noun_verb_noun', rule)

        q, a = ("does this shirt have a design on it", "no")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('have', rule)

        q, a = ("do the animals seem healthy", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('look_adj', rule)

        q, a = ("is the dome gold", "no it's black")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('noun_noun', rule)

        q, a = ("what is the boy wearing", "a blue shirt and jeans")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('what_is', rule)

        q, a = ("is the water calm or rough", "calm")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('adj_or_adj', rule)

        q, a = ("is the phone a smartphone", "yes")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('is_np_np', rule)

        q, a, coref_q = ('is she sad', 'no', 'is the woman in the party sad')
        _, rule = generate_proposition(q, a, coref_q, a)
        self.assertEqual('verb_pron_adj', rule)

        q, a, coref_q = ("are they planting a tree", "yes",
                         "are the little children planting a tree")
        _, rule = generate_proposition(q, a, coref_q, a)
        self.assertEqual('verb_pron_obj', rule)

        q, a, coref_q = ("is he on a sidewalk", "no he's in the street",
                         "is the man on a sidewalk")
        _, rule = generate_proposition(q, a, coref_q, a)
        self.assertEqual('pron_prep', rule)

        q, a = ("does it look like a forest", "no")
        _, rule = generate_proposition(q, a, q, a)
        self.assertEqual('look_like', rule)

    def test_caption(self):

        caption = "'a colorful bird is sitting on a small branch"
        props = ['one can see a colorful bird.',
                 'one cannot see a colorful bird.',
                 'there is a bird.', 'there is no bird.',
                 'one can see a small branch.',
                 'one cannot see a small branch.',
                 'there is a branch.', 'there is no branch.']

        for (e, c), _ in propositions_from_caption(caption):
            self.assertIn(e, props)
            self.assertIn(c, props)

    def test_what_color(self):
        #  5581
        q, a = ("what color are the sheep", "dirty white")
        entailment = "the sheep are white."
        contradiction = "the sheep are not white."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5584
        q, a = ("what color is the water", "blue")
        entailment = "the water is blue."
        contradiction = "the water is not blue."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5600
        q, a = ("what color is the street sign", "red white and black")
        entailment = "the street sign is red, white and black."
        contradiction = "the street sign is not red, white and black."
        (e, c), _ = what_color(q, a, q)
        self.assertCountEqual(e, entailment)
        self.assertCountEqual(c, contradiction)
        #  5622
        q, a = ("what color are bears", "brown, green and white")
        entailment = "the bears are brown, green and white."
        contradiction = "the bears are not brown, green and white."
        (e, c), _ = what_color(q, a, q)
        self.assertCountEqual(e, entailment)
        self.assertCountEqual(c, contradiction)
        #  5632
        q, a = ("what color is the fence", "wood posts and wire i think")
        entailment = "the fence is wood."
        contradiction = "the fence is not wood."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5644
        q, a = ("what color is the car", "looks like dark gray")
        entailment = "the car is dark gray."
        contradiction = "the car is not dark gray."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5660
        q, a = ("what color is the bowl", "brown")
        entailment = "the bowl is brown."
        contradiction = "the bowl is not brown."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5663
        q, a = ("what color is the table", "light wood")
        entailment = "the table is light wood."
        contradiction = "the table is not light wood."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5688
        q, a = ("what color are the buildings", "dingy red")
        entailment = "the buildings are red."
        contradiction = "the buildings are not red."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5866
        q, a = ("what color are the blankets on the bed",
                "it's a print that looks beige and brown")
        entailment = "the blankets on the bed are beige and brown."
        contradiction = "the blankets on the bed are not beige and brown."
        (e, c), _ = what_color(q, a, q)
        # color order can change because it uses sets, so we check that
        # the Counter of characters match.
        self.assertEqual(Counter(e), Counter(entailment))
        self.assertEqual(Counter(c), Counter(contradiction))
        #  5869
        q, a = ("what color is the decor in the room", "beige")
        entailment = "the decor in the room is beige."
        contradiction = "the decor in the room is not beige."
        (e, c), _ = what_color(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5810
        q, a = ("what color is the bird", "brown with some orange in the wings")
        entailment = "the bird is brown and orange."
        contradiction = "the bird is not brown and orange."
        (e, c), _ = what_color(q, a, q)
        self.assertCountEqual(e, entailment)
        self.assertCountEqual(c, contradiction)

        q, a = ("what color is", "red")
        output = what_color(q, a, q, no_verb=True)
        self.assertEqual(output, [])

    def test_what_color_no_verb(self):
        #  5573
        q, a = ("what color walls", "white")
        entailment = "the walls are white."
        contradiction = "the walls are not white."
        (e, c), _ = what_color(q, a, q, no_verb=True)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5574
        q, a = ("what color sink", "white")
        entailment = "the sink is white."
        contradiction = "the sink is not white."
        (e, c), _ = what_color(q, a, q, no_verb=True)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5764
        q, a = ("what color uniforms",
                "it is hard to tell, twilight, maybe dark blue")
        entailment = "the uniforms are dark blue."
        contradiction = "the uniforms are not dark blue."
        (e, c), _ = what_color(q, a, q, no_verb=True)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  5572
        q, a = ("what color cabinets", "no cabinets")
        output = what_color(q, a, q, no_verb=True)
        self.assertEqual(output, [])

    def test_can_you_see(self):
        #  8907
        q, a = ("can you see any signage at the train station", "no")
        entailment = "one cannot see any signage at the train station."
        contradiction = "one can see a signage at the train station."
        (e, c), _ = can_you_see(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  8988
        q, a = ("can you see a sink", "yes")
        entailment = "one can see a sink."
        contradiction = "one cannot see any sink."
        (e, c), _ = can_you_see(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  23028
        q, a = ("do you see any buildings", "no")
        entailment = "one cannot see any buildings."
        contradiction = "one can see buildings."
        (e, c), _ = can_you_see(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  23022
        q, a = ("do you see a table", "yes i do")
        entailment = "one can see a table."
        contradiction = "one cannot see any table."
        (e, c), _ = can_you_see(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        #  8981
        q, a = ("can you see the floor", "yes")
        entailment = "one can see a floor."
        contradiction = "one cannot see any floor."
        (e, c), _ = can_you_see(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_there(self):
        # 528
        q, a = ("are there any skydivers coming out of the plane", "no")
        entailment = "there are no skydivers coming out of the plane."
        contradiction = "there are skydivers coming out of the plane."
        (e, c), _ = there(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 563
        q, a = ("are there people in the water",
                "there is a person in the water on a jet ski")
        output = there(q, a, q)
        self.assertEqual([], output)
        # 566
        q, a = ("are there any animals in the image", "no")
        entailment = "there are no animals in the image."
        contradiction = "there are animals in the image."
        (e, c), _ = there(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 635
        q, a = ("are there any street lights", "yes")
        entailment = "there are street lights."
        contradiction = "there are no street lights."
        (e, c), _ = there(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 636
        q, a = ("are there buildings", "yes")
        entailment = "there are buildings."
        contradiction = "there are no buildings."
        (e, c), _ = there(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 943
        q, a = ("is there grass", "yes")
        entailment = "there is grass."
        contradiction = "there is no grass."
        (e, c), _ = there(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 973
        q, a = ("is there a sink", "not in this view")
        entailment = "there is no sink."
        contradiction = "there is a sink."
        (e, c), _ = there(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 1065
        q, a = ("is there any coffee", "no coffee is visible")
        entailment = "there is no coffee."
        contradiction = "there is coffee."
        (e, c), _ = there(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_any(self):
        # 957
        q, a = ("any trees", "no, there are no trees")
        entailment = "there are no trees."
        contradiction = "there are trees."
        (e, c), _ = any_(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 969
        q, a = ("any person", "no")
        entailment = "there is no person."
        contradiction = "there is a person."
        (e, c), _ = any_(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 1097
        q, a = ("any lakes nearby", "no")
        entailment = "there are no lakes nearby."
        contradiction = "there are lakes nearby."
        (e, c), _ = any_(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 1329
        q, a = ("any buildings in sight", "yes")
        entailment = "there are buildings in sight."
        contradiction = "there are no buildings in sight."
        (e, c), _ = any_(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 1247
        q, a = ("any lettering on their uniforms", "yes")
        entailment = "there is a lettering on their uniforms."
        contradiction = "there is no lettering on their uniforms."
        (e, c), _ = any_(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_weather(self):
        # 15926
        q, a = ("how is the weather", "clear")
        entailment = "the weather is clear."
        contradiction = "the weather is not clear."
        (e, c), _ = weather(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 15987
        q, a = ("how is weather looking like", "sunny")
        entailment = "the weather is sunny."
        contradiction = "the weather is not sunny."
        (e, c), _ = weather(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 17554
        q, a = ("what does the weather look like", "sunny and cold")
        entailment = "the weather is sunny and cold."
        contradiction = "the weather is not sunny and cold."
        (e, c), _ = weather(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 25187
        q, a = ("what is the weather like", "the weather is beautiful")
        entailment = "the weather is beautiful."
        contradiction = "the weather is not beautiful."
        (e, c), _ = weather(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_what_kind(self):
        # 2741
        q, a = ("what kind of birds are they", "seagulls")
        entailment = "the birds are seagulls."
        contradiction = "the birds are not seagulls."
        (e, c), _ = what_kind(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 3711
        q, a = ("what kind of food", "asian")
        entailment = "the food is asian."
        contradiction = "the food is not asian."
        (e, c), _ = what_kind(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # # 4159
        q, a = ("what kind of flooring", "tile")
        entailment = "the flooring is tile."
        contradiction = "the flooring is not tile."
        (e, c), _ = what_kind(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 5621
        q, a = ("what kind of animals", "teddy bears")
        entailment = "the animals are teddy bears."
        contradiction = "the animals are not teddy bears."
        (e, c), _ = what_kind(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_is_it(self):
        # 2284
        q, a = ("is it an ornate frame", "no")
        entailment = "there is not an ornate frame."
        contradiction = "there is an ornate frame."
        (e, c), _ = is_it(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 2454
        q, a = ("is it a boy", "yes, it is")
        entailment = "there is a boy."
        contradiction = "there is not a boy."
        (e, c), _ = is_it(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 2483
        q, a = ("is it a single use bathroom", "yes")
        entailment = "there is a single use bathroom."
        contradiction = "there is not a single use bathroom."
        (e, c), _ = is_it(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 3370
        q, a = ("is it a baby", "no they are large")
        entailment = "there is not a baby."
        contradiction = "there is a baby."
        (e, c), _ = is_it(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 3560
        q, a = ("is it an adult dog", "yes")
        entailment = "there is an adult dog."
        contradiction = "there is not an adult dog."
        (e, c), _ = is_it(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_is_inside(self):
        # 20828
        q, a = ("is it outside", "no")
        entailment = "the image is not outside."
        contradiction = "the image is outside."
        (e, c), _ = is_inside(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 21570
        q, a = ("is this outdoors", "yes")
        entailment = "the image is outdoors."
        contradiction = "the image is not outdoors."
        (e, c), _ = is_inside(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 23350
        q, a = ("is this outside", "no, it's inside")
        entailment = "the image is not outside."
        contradiction = "the image is outside."
        (e, c), _ = is_inside(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 26171
        q, a = ("is this inside", "yes")
        entailment = "the image is inside."
        contradiction = "the image is not inside."
        (e, c), _ = is_inside(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_in_answer(self):
        _, a = ("", "inside")
        entailment = "the image is inside."
        contradiction = "the image is not inside."
        (e, c), _ = in_answer(a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_out_answer(self):
        _, a = ("", "outdoors")
        entailment = "the image is outdoors."
        contradiction = "the image is not outdoors."
        (e, c), _ = out_answer(a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_sunny(self):
        q, a = ("is it sunny", "yes")
        entailment = "it is sunny."
        contradiction = "it is not sunny."
        (e, c), _ = sunny(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_cloudy(self):
        q, a = ("is it cloudy", "no")
        entailment = "it is not cloudy."
        contradiction = "it is cloudy."
        (e, c), _ = cloudy(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_weather_answer(self):
        _, a = ("", "overcast")
        entailment = "it is overcast."
        contradiction = "it is not overcast."
        (e, c), _ = weather_answer(a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_daytime(self):
        q, a = ("is it daytime", "yes")
        entailment = "it is daytime."
        contradiction = "it is not daytime."
        (e, c), _ = daytime(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_daytime_answer(self):
        _, a = ("", "daytime")
        entailment = "it is daytime."
        contradiction = "it is not daytime."
        (e, c), _ = daytime_answer(a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_image_in_color(self):
        q, a = ("is the photo in color", "yes")
        entailment = "the photo is in color."
        contradiction = "the photo is not in color."
        (e, c), _ = image_in_color(q, a)
        e = e.replace('image', 'photo').replace('picture', 'photo')
        c = c.replace('image', 'photo').replace('picture', 'photo')
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_person_answer(self):
        _, a = ("", "female")
        entailment = "there is a female."
        contradiction = "there is no female."
        (e, c), _ = person_answer(a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_no_people(self):
        _, a = ("", "no people")
        entailment = "there are no people."
        contradiction = "there are people."
        (e, c), _ = no_people(a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_is_noun_adj(self):
        # 7188
        q, a = ("is sky visible", "yes")
        entailment = "the sky is visible."
        contradiction = "the sky is not visible."
        (e, c), _ = is_noun_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 7340
        q, a = ("is the train long", "no, it's short")
        entailment = "the train is not long."
        contradiction = "the train is long."
        (e, c), _ = is_noun_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 7386
        q, a = ("is the ground visible", "no")
        entailment = "the ground is not visible."
        contradiction = "the ground is visible."
        (e, c), _ = is_noun_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 7392
        q, a = ("is the wave really large", "yes")
        entailment = "the wave is really large."
        contradiction = "the wave is not really large."
        (e, c), _ = is_noun_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 7480
        q, a = ("is the woman alone", "yes")
        entailment = "the woman is alone."
        contradiction = "the woman is not alone."
        (e, c), _ = is_noun_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 7749
        q, a = ("is the room clean", "no")
        entailment = "the room is not clean."
        contradiction = "the room is clean."
        (e, c), _ = is_noun_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_noun_(self):
         # 117023
        q, a = ("trees", "yes")
        entailment = "there are trees."
        contradiction = "there are no trees."
        (e, c), _ = noun_(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 117242
        q, a = ("gloves", "yes")
        entailment = "there are gloves."
        contradiction = "there are no gloves."
        (e, c), _ = noun_(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 124597
        q, a = ("animals", "no")
        entailment = "there are no animals."
        contradiction = "there are animals."
        (e, c), _ = noun_(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 126271
        q, a = ("window", "no, i don't see any windows")
        entailment = "there is no window."
        contradiction = "there is a window."
        (e, c), _ = noun_(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_noun_prep_noun(self):
        # 23342
        q, a = ("is the catcher in a crouch", "yes")
        entailment = "the catcher is in a crouch."
        contradiction = "the catcher is not in a crouch."
        (e, c), _ = noun_prep_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 23610
        q, a = ("is the tie on a person",
                "yes but i can't see anything but part of a neck")
        entailment = "the tie is on a person."
        contradiction = "the tie is not on a person."
        (e, c), _ = noun_prep_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 24615
        q, a = ("are the owners in the picture", "no")
        entailment = "the owners are not in the picture."
        contradiction = "the owners are in the picture."
        (e, c), _ = noun_prep_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 24958
        q, a = ("is the kit in the air", "no")
        entailment = "the kit is not in the air."
        contradiction = "the kit is in the air."
        (e, c), _ = noun_prep_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 71842
        q, a = ("is the woman in glasses", "yes")
        entailment = "the woman is in glasses."
        contradiction = "the woman is not in glasses."
        (e, c), _ = noun_prep_noun(q, a, det=False)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 72051
        q, a = ("is the cake in slices", "yes")
        entailment = "the cake is in slices."
        contradiction = "the cake is not in slices."
        (e, c), _ = noun_prep_noun(q, a, det=False)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_noun_verb(self):
        # 65307
        q, a = ("is the window covered", "it is halfway covered")
        entailment = "the window is covered."
        contradiction = "the window is not covered."
        (e, c), _ = noun_verb(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 65532
        q, a = ("is the motorcycle moving", "no")
        entailment = "the motorcycle is not moving."
        contradiction = "the motorcycle is moving."
        (e, c), _ = noun_verb(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 65901
        q, a = ("are the vegetables cooked", "yes")
        entailment = "the vegetables are cooked."
        contradiction = "the vegetables are not cooked."
        (e, c), _ = noun_verb(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 66803
        q, a = ("is the display covered", "yes it is")
        entailment = "the display is covered."
        contradiction = "the display is not covered."
        (e, c), _ = noun_verb(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 66937
        q, a = ("is the motorcycle moving", "no parked")
        entailment = "the motorcycle is not moving."
        contradiction = "the motorcycle is moving."
        (e, c), _ = noun_verb(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 67498
        q, a = ("is the cat sleeping", "no it's awake")
        entailment = "the cat is not sleeping."
        contradiction = "the cat is sleeping."
        (e, c), _ = noun_verb(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def noun_verb_noun(self):
        # 148336
        q, a = ("is the man throwing the frisbee", "yes he is throwing it")
        entailment = "the man is throwing the frisbee. "
        contradiction = "the man is not throwing the frisbee. "
        (e, c), _ = noun_verb_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 148468
        q, a = ("is the man wearing a hat", "no")
        entailment = "the man is not wearing a hat."
        contradiction = "the man is wearing a hat."
        (e, c), _ = noun_verb_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 148463
        q, a = ("is the man wearing a top", "yes")
        entailment = "the man is wearing a top."
        contradiction = "the man is not wearing a top."
        (e, c), _ = noun_verb_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 148803
        q, a = ("are the horses facing the camera", "no")
        entailment = "the horses are not facing the camera."
        contradiction = "the horses are facing the camera."
        (e, c), _ = noun_verb_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 244465
        q, a = ("is the girl wearing glasses", "no")
        entailment = "the girl is not wearing glasses."
        contradiction = "the girl is wearing glasses."
        (e, c), _ = noun_verb_noun(q, a, det=False)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 247818
        q, a = ("are the riders wearing armor", "yes")
        entailment = "the riders are wearing armor."
        contradiction = "the riders are not wearing armor."
        (e, c), _ = noun_verb_noun(q, a, det=False)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_have(self):
        # 71159
        q, a = ("does the man have a hat on", "no")
        entailment = "the man does not have a hat on."
        contradiction = "the man has a hat on."
        (e, c), _ = have(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 71448
        q, a = ("does this shirt have a design on it", "no")
        entailment = "this shirt does not have a design on it."
        contradiction = "this shirt has a design on it."
        (e, c), _ = have(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 71764
        q, a = ("does the twig have branches", "yes")
        entailment = "the twig has branches."
        contradiction = "the twig does not have branches."
        (e, c), _ = have(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_look_adj(self):
        # 79269
        q, a = ("do the people look happy", "no")
        entailment = "the people do not look happy."
        contradiction = "the people look happy."
        (e, c), _ = look_adj(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 79343
        q, a = ("does the sign look new", "yes")
        entailment = "the sign looks new."
        contradiction = "the sign does not look new."
        (e, c), _ = look_adj(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 79534
        q, a = ("do the animals seem healthy", "yes")
        entailment = "the animals seem healthy."
        contradiction = "the animals do not seem healthy."
        (e, c), _ = look_adj(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 81900
        q, a = ("does the man look dirty", "no")
        entailment = "the man does not look dirty."
        contradiction = "the man looks dirty."
        (e, c), _ = look_adj(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_noun_noun(self):
        # 119217
        q, a = ("is the rock real", "yes")
        entailment = "the rock is real."
        contradiction = "the rock is not real."
        (e, c), _ = noun_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 119484
        q, a = ("is the dome gold", "no it's black")
        entailment = "the dome is not gold."
        contradiction = "the dome is gold."
        (e, c), _ = noun_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 119602
        q, a = ("is the table wood", "yes it is wood")
        entailment = "the table is wood."
        contradiction = "the table is not wood."
        (e, c), _ = noun_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 120968
        q, a = ("is the bird eating", "no")
        entailment = "the bird is not eating."
        contradiction = "the bird is eating."
        (e, c), _ = noun_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 121347
        q, a = ("are the sheep grazing", "no")
        entailment = "the sheep are not grazing."
        contradiction = "the sheep are grazing."
        (e, c), _ = noun_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 121402
        q, a = ("are the sheep wooly", "yes")
        entailment = "the sheep are wooly."
        contradiction = "the sheep are not wooly."
        (e, c), _ = noun_noun(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_what_is(self):
        # 197347
        q, a = ("what are the people wearing", "snow jackets, hats, and boots")
        entailment = "the people are wearing snow jackets, hats, and boots."
        contradiction = "the people are not wearing snow jackets, hats, and boots."
        (e, c), _ = what_is(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 198573
        q, a = ("what is the man doing", "holding his hands together")
        entailment = "the man is holding his hands together."
        contradiction = "the man is not holding his hands together."
        (e, c), _ = what_is(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 201348
        q, a = ("what is the boy wearing", "a blue shirt and jeans")
        entailment = "the boy is wearing a blue shirt and jeans."
        contradiction = "the boy is not wearing a blue shirt and jeans."
        (e, c), _ = what_is(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 202039
        q, a = ("what is the giraffe doing", "looking behind himself")
        entailment = "the giraffe is looking behind himself."
        contradiction = "the giraffe is not looking behind himself."
        (e, c), _ = what_is(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_adj_or_adj(self):
        # 194894
        q, a = ("is the dog big or small", "it's kind of small")
        entailment = "the dog is small."
        contradiction = "the dog is not small."
        (e, c), _ = adj_or_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 195048
        q, a = ("is the building old or new", "looks new")
        entailment = "the building is new."
        contradiction = "the building is not new."
        (e, c), _ = adj_or_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 196742
        q, a = ("is the water calm or rough", "calm")
        entailment = "the water is calm."
        contradiction = "the water is not calm."
        (e, c), _ = adj_or_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 196983
        q, a = ("is the plate round or square", "round")
        entailment = "the plate is round."
        contradiction = "the plate is not round."
        (e, c), _ = adj_or_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 198798
        q, a = ("is the toilet open or closed", "the toilet is closed")
        entailment = "the toilet is closed."
        contradiction = "the toilet is not closed."
        (e, c), _ = adj_or_adj(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_is_np_np(self):
        # 159229
        q, a = ("is the zebra a baby", "no, the zebra isn't a baby")
        entailment = "the zebra is not a baby."
        contradiction = "the zebra is a baby."
        (e, c), _ = is_np_np(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 159370
        q, a = ("is the phone a smartphone", "yes")
        entailment = "the phone is a smartphone."
        contradiction = "the phone is not a smartphone."
        (e, c), _ = is_np_np(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 160628
        q, a = ("is the sign a large billboard", "no")
        entailment = "the sign is not a large billboard."
        contradiction = "the sign is a large billboard."
        (e, c), _ = is_np_np(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 160906
        q, a = ("is the water a river", "no ocean")
        entailment = "the water is not a river."
        contradiction = "the water is a river."
        (e, c), _ = is_np_np(q, a)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_verb_pron_adj(self):
        q, a, coref_q = ('is he old', 'yes', 'is the man in the boat old')
        entailment = "the man in the boat is old."
        contradiction = "the man in the boat is not old."
        (e, c), _ = verb_pron_adj(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

        q, a, coref_q = ('is she sad', 'no', 'is the woman in the party sad')
        entailment = "the woman in the party is not sad."
        contradiction = "the woman in the party is sad."
        (e, c), _ = verb_pron_adj(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_verb_pron_obj(self):
        # 72201
        q, a, coref_q = ("is he wearing a wetsuit", "no, he is not",
                         'is the man wearing a wetsuit')
        entailment = "the man is not wearing a wetsuit."
        contradiction = "the man is wearing a wetsuit."
        (e, c), _ = verb_pron_obj(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

        q, a, coref_q = ("are they planting a tree", "yes",
                         "are the little children planting a tree")
        entailment = "the little children are planting a tree."
        contradiction = "the little children are not planting a tree."
        (e, c), _ = verb_pron_obj(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_pron_prep(self):
        # 141272
        q, a, coref_q = ("is he on a sidewalk", "no he's in the street",
                         "is the man on a sidewalk")
        entailment = "the man is not on a sidewalk."
        contradiction = "the man is on a sidewalk."
        (e, c), _ = pron_prep(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 140972
        q, a, coref_q = ("is he in a uniform", "yes red and white",
                         "is the young soldier in a uniform")
        entailment = "the young soldier is in a uniform."
        contradiction = "the young soldier is not in a uniform."
        (e, c), _ = pron_prep(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_look_like(self):
        # 400722
        q, a = ("does it look like a public restroom", "yes")
        entailment = "it looks like a public restroom."
        contradiction = "it does not look like a public restroom."
        (e, c), _ = look_like(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 403986
        q, a = ("does it look like a big city", "yes very")
        entailment = "it looks like a big city."
        contradiction = "it does not look like a big city."
        (e, c), _ = look_like(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

        q, a = ("does it look like a forest", "no")
        entailment = "it does not look like a forest."
        contradiction = "it looks like a forest."
        (e, c), _ = look_like(q, a, q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 421122
        q, a, coref_q = ("do they look like a couple", "no",
                         "do the man and the woman look like a couple")
        entailment = "the man and the woman do not look like a couple."
        contradiction = "the man and the woman look like a couple."
        (e, c), _ = look_like(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)
        # 421125
        q, a, coref_q = ("do they look like family", "no",
                         "do the people in the picture look like family")
        entailment = "the people in the picture do not look like family."
        contradiction = "the people in the picture look like family."
        (e, c), _ = look_like(q, a, coref_q)
        self.assertEqual(e, entailment)
        self.assertEqual(c, contradiction)

    def test_remove_functions(self):
        prefix = 'can you see if '
        sentence = 'can you see if the people far away are smiling'
        suffix = ' are smiling'
        string_1 = removeprefix(sentence, prefix)
        string_2 = removesuffix(string_1, suffix)
        self.assertEqual(string_2, 'the people far away')


if __name__ == '__main__':
    unittest.main()
