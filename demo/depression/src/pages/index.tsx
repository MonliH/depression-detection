import {
  Box,
  Button,
  Heading,
  HStack,
  Input,
  InputGroup,
  InputLeftElement,
  Text,
  VStack,
} from "@chakra-ui/react";
import Head from "next/head";
import { useState } from "react";
import { Search } from "react-feather";

interface Comment {
  text: string;
  subreddit: string;
}

function round(num: number) {
  return Math.round((num + Number.EPSILON) * 100) / 100;
}

export default function Home() {
  const [username, setUsername] = useState<string>("");
  const [comments, setComments] = useState<Comment[]>([]);
  const [confidence, setConfidence] = useState<null | [number, number]>(null);

  const getComments = () => {
    fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/get-posts`, {
      body: JSON.stringify({ username }),
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((data) => {
        setComments(data);
        setConfidence(null);
      });
  };

  const predict = () => {
    fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/predict`, {
      body: JSON.stringify({ posts: comments }),
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((data) => {
        setConfidence(data);
      });
  };

  return (
    <>
      <Head>
        <title>Depression Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        <Box height="100vh" w="100%" overflow="hidden">
          <Heading>Depression Detection</Heading>
          <HStack>
            <Input
              placeholder="Username"
              onChange={(e) => setUsername(e.target.value)}
            />
            <Button onClick={getComments}>Get Comments</Button>
          </HStack>
          <HStack w="100%" h="100%">
            <VStack alignItems="baseline" w="50%" h="100%">
              <InputGroup flexGrow={0}>
                <InputLeftElement pointerEvents={"none"}>
                  <Search />
                </InputLeftElement>
                <Input type="text" placeholder="Search" />
              </InputGroup>
              <Box
                p="4"
                borderRight="1px solid"
                borderRightColor="gray.200"
                flexGrow={1}
                w="100%"
                overflow="scroll"
              >
                <VStack alignItems="baseline">
                  {comments.map((comment, i) => (
                    <Box
                      key={i}
                      p="3"
                      border="1px solid"
                      borderColor="gray.200"
                      borderRadius="5"
                      w="100%"
                    >
                      <Text fontSize="xs" fontWeight="bold">
                        /r/{comment.subreddit}
                      </Text>
                      <Text mt="1">{comment.text}</Text>
                    </Box>
                  ))}
                </VStack>
              </Box>
            </VStack>
            <Box w='100%' p="5">
              <Button onClick={predict} mb="4">
                Predict
              </Button>
              {confidence && (
                // show bar graph
                <Box w="100%">
                  <HStack>
                  <Text w="110px">Not Depressed</Text>
                  <Box w={`${70*confidence[0]}%`} h="10px" bg="blue.500" />
                  <Text>{round(confidence[0])}</Text>
                  </HStack>
                  <HStack>
                  <Text w="110px">Depressed</Text>
                  <Box w={`${70*confidence[1]}%`} h="10px" bg="red.500" />
                  <Text>{round(confidence[1])}</Text>
                  </HStack>
                </Box>
              )}
            </Box>
          </HStack>
        </Box>
      </main>
    </>
  );
}
