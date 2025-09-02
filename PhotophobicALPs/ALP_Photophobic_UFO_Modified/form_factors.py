from object_library import all_form_factors, FormFactor

from function_library import complexconjugate, re, im, csc, sec, acsc, asec
# incoming nucleon id 2, outgoing nucleon id 4

# t = (- 2*MNul**2 - 2*P(-1,1)*P(-1,2))

AAA = FormFactor(name = 'AAA',
                 type = 'real',
                 value = '((ZNuc**2*(aval**2*(- 2*MNuc**2 - 2*P(-1,1)*P(-1,2))/(1+aval**2*(- 2*MNuc**2 - 2*P(-1,1)*P(-1,2))))**2*(1/(1+(- 2*MNuc**2 - 2*P(-1,1)*P(-1,2))/dval))**2)+ZNuc*(apval**2*(- 2*MNuc**2 - 2*P(-1,1)*P(-1,2))/(1+apval**2*(- 2*MNuc**2 - 2*P(-1,1)*P(-1,2))))**2*((1+(- 2*MNuc**2 - 2*P(-1,1)*P(-1,2))*inelastic1)/(1+(- 2*MNuc**2 - 2*P(-1,1)*P(-1,2))*inelastic2)**4)**2)**0.5'
                 )
