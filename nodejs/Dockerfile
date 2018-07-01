FROM node

COPY *.js /app/
COPY *.json /app/

ENV http_proxy http://proxy.kancelar.seznam.cz:3128 
RUN npm install supervisor -g

EXPOSE 8000

WORKDIR /app/

CMD supervisor server.js
#CMD npm start 
