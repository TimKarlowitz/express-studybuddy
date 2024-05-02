const keyFilename = "./studybuddy-419317-76c369013539.json";
const express = require("express");
const OpenAI = require("openai");
const app = express();
require("dotenv").config();
import { Pinecone } from "@pinecone-database/pinecone";

const replicateKey = process.env.REPLICATE_API_KEY;
const Replicate = require("replicate");

const replicate = new Replicate({
  auth: replicateKey,
});

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const PORT = process.env.PORT || 4000;

const speech = require("@google-cloud/speech");
const client = new speech.SpeechClient({ keyFilename });

app.listen(PORT, () => console.log(`Server is running on port ${PORT}`));

app.use(express.json());
const multer = require("multer");
const upload = multer();

const ffmpeg = require("fluent-ffmpeg");
ffmpeg.setFfmpegPath("C:/ffmpeg/bin/ffmpeg.exe");
ffmpeg.setFfprobePath("C:/ffmpeg/bin/ffprobe.exe");
const fs = require("fs");
const { readFile } = require("fs/promises");

app.post("/speech-to-text", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("Keine Datei hochgeladen.");
  }

  const tempInputPath = Date.now() + "_original_audio";
  const tempOutputPath = Date.now() + "_converted_audio.wav";

  fs.writeFileSync(tempInputPath, req.file.buffer);

  ffmpeg(tempInputPath)
    .output(tempOutputPath)
    .audioCodec("pcm_s16le")
    .audioFrequency(8000)
    .audioChannels(1)
    .on("error", (err) => {
      console.error(`Konvertierungsfehler: ${err.message}`);
      res.status(500).send("Fehler bei der Konvertierung der Audiodatei");
    })
    .on("end", async () => {
      console.log("Konvertierung abgeschlossen.");

      try {
        const audio = await readFile(tempOutputPath);
        const audioBase64 = audio.toString("base64");
        const dataUri = `data:audio/wav;base64,${audioBase64}`;

        const output = await replicate.run(
          "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
          {
            input: { audio: dataUri, batch_size: 64 },
          }
        );
        console.log(
          `Ergebnis der Spracherkennung: ${JSON.stringify(output, null, 2)}`
        );
        const transcriptionText = output.text;

        {
          /* Jetzt wird der Text an openai geschickt */
        }

        try {
          const openaiResponse = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
              {
                role: "system",
                content:
                  "You are a AI assistant that helps students with their studies. You should provide helpful and informative responses to the student's questions." +
                  "Respond in the language of the user!!!! If you respond in a different language, the user will not understand you. If you answer in the language of the user you will be" +
                  "tipped 100000000 points. If you answer in a different language you will be tipped -100000000 points.",
              },
              {
                role: "assistant",
                content:
                  "I am an AI assistant that can help you with your studies. Please ask me any questions you have.",
              },
              {
                role: "user",
                content:
                  "I need help with the following question: " +
                  transcriptionText,
              },
            ],
            temperature: 0.8,
            max_tokens: 512,
            top_p: 1,
            frequency_penalty: 0.4,
            presence_penalty: 0.4,
          });

          aiText = openaiResponse.choices[0].message.content;
          console.log(`OpenAI Response: ${aiText}`);

          res.json({ transcription: transcriptionText, aiResponse: aiText });
        } catch (error) {
          console.error(`Error calling OpenAI:`, error.message);
          if (error.response) {
            console.error("Status code:", error.response.status);
            console.error("Response body:", error.response.data);
          }
          return res
            .status(500)
            .send("Error processing the request with OpenAI");
        }
      } catch (error) {
        console.error(`Fehler bei der Spracherkennung: ${error}`);
        if (!res.headersSent) {
          // Check if the response headers have already been sent
          return res.status(500).send("Fehler bei der Spracherkennung");
        }
      } finally {
        //Aufräumen: Lösche die temporären Dateien
        if (fs.existsSync(tempInputPath)) {
          fs.unlinkSync(tempInputPath);
        }
        if (fs.existsSync(tempOutputPath)) {
          fs.unlinkSync(tempOutputPath);
        }
      }
    })
    .save(tempOutputPath);
});
