// Import von externen Bibliotheken
import express from "express";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import Replicate from "replicate";
import { SpeechClient } from "@google-cloud/speech";
import multer from "multer";
import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import { readFile } from "fs/promises";

import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(express.json());
const PORT = process.env.PORT || 4000;

const client = new SpeechClient({
  keyFilename: "./studybuddy-419317-76c369013539.json",
});
const replicate = new Replicate({ auth: process.env.REPLICATE_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

const upload = multer();
ffmpeg.setFfmpegPath("C:/ffmpeg/bin/ffmpeg.exe");
ffmpeg.setFfprobePath("C:/ffmpeg/bin/ffprobe.exe");

app.listen(PORT, () => console.log(`Server is running on port ${PORT}`));

app.post("/speech-to-text", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("Keine Datei hochgeladen.");
  }

  const tempInputPath = Date.now() + "_original_audio";
  const tempOutputPath = Date.now() + "_converted_audio.wav";
  fs.writeFileSync(tempInputPath, req.file.buffer);

  try {
    await convertAudio(tempInputPath, tempOutputPath);
    const transcriptionText = await transcribeAudio(tempOutputPath);
    const aiText = await queryOpenAI(transcriptionText);
    res.json({ transcription: transcriptionText, aiResponse: aiText });
  } catch (error) {
    console.error(error);
    res.status(500).send(error.message);
  } finally {
    cleanUpFiles([tempInputPath, tempOutputPath]);
  }
});

async function convertAudio(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .output(outputPath)
      .audioCodec("pcm_s16le")
      .audioFrequency(8000)
      .audioChannels(1)
      .on("error", (err) => reject(`Konvertierungsfehler: ${err.message}`))
      .on("end", () => {
        console.log("Konvertierung abgeschlossen.");
        resolve();
      })
      .save(outputPath);
  });
}

async function transcribeAudio(filePath) {
  const audio = await readFile(filePath);
  const audioBase64 = audio.toString("base64");
  const dataUri = `data:audio/wav;base64,${audioBase64}`;

  const output = await replicate.run(
    "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
    { input: { audio: dataUri, batch_size: 64 } }
  );

  console.log(
    `Ergebnis der Spracherkennung: ${JSON.stringify(output, null, 2)}`
  );
  return output.text;
}

async function queryOpenAI(transcriptionText) {
  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {
        role: "system",
        content:
          "You are a AI assistant that helps students with their studies. You should provide helpful and informative responses to the student's questions. Respond in the language of the user!!!! If you respond in a different language, the user will not understand you. If you answer in the language of the user you will be tipped 100000000 points. If you answer in a different language you will be tipped -100000000 points.",
      },
      {
        role: "assistant",
        content:
          "I am an AI assistant that can help you with your studies. Please ask me any questions you have.",
      },
      {
        role: "user",
        content:
          "I need help with the following question: " + transcriptionText,
      },
    ],
    temperature: 0.8,
    max_tokens: 512,
    top_p: 1,
    frequency_penalty: 0.4,
    presence_penalty: 0.4,
  });

  return response.choices[0].message.content;
}

function cleanUpFiles(paths) {
  paths.forEach((path) => {
    if (fs.existsSync(path)) {
      fs.unlinkSync(path);
    }
  });
}
