# ImageMatting-Introduction_to_Visual_Computing
Solve image mapping problem with triangulation equation using python/numpy /opencv

The file Triangulation_mapping.py deal with all logics and algorithms. 
It takes five import pictures with the same size and output one alpha imgae, one color image and a composite image which matting the background to foreground object. 

Strength:
Triangulating matting equation could deal with different level of transparency from bottle to paper pretty good and handle different textures very well.In addition, Triangulating matting equation does not have any limit in terms of foreground object’s color.
Limits:
Triangulating matting equation does not bear any subtle difference in input pictures. And It would regard some light reflections and mirrored image as the object’s own property.
