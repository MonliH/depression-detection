import {
  Box,
  Button,
  Divider,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Heading,
  HStack,
  Input,
  InputGroup,
  InputLeftElement,
  InputRightElement,
  Text,
  Tooltip,
  VStack,
} from "@chakra-ui/react";
import Head from "next/head";
import { useCallback, useEffect, useRef, useState } from "react";
import { Search } from "react-feather";
import elasticlunr from "elasticlunr";
import debounce from "lodash.debounce";

interface Comment {
  text: string;
  subreddit: string;
  id: number;
}

const SUBREDDITS_RELATED_TO_DEPRESSION = new Set([
  "Anxiety",
  "anxietyhelp",
  "anxietysuccess",
  "anxietysupporters",
  "CPTSD",
  "dpdr",
  "HealthAnxiety",
  "OCD",
  "PanicAttack",
  "Phobia",
  "pureo",
  "ptsd",
  "socialanxiety",
  "OCD",
  "depression",
  "depressed",
  "depression_help",
  "depressionregiments",
  "DepressionJournals",
  "DepressionRecovery",
  "dysthymia",
  "AnxietyDepression",
  "adhd_anxiety",
  "ADHD",
  "AdultADHDSupportGroup",
  "ashhd",
].map((s) => s.toLowerCase()));

function round(num: number) {
  return Math.round((num + Number.EPSILON) * 100) / 100;
}

export default function Home() {
  const [username, setUsername] = useState<string>("");
  const [currentUser, setCurrentUser] = useState<string>("");
  const [comments, setComments] = useState<Comment[]>([]);
  const [confidence, setConfidence] = useState<null | [number, number]>(null);
  const [loadingComments, setLoadingComments] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>("");

  const [formError, setFormError] = useState<string | null>(null);
  const search = useRef<elasticlunr.Index<Comment>>();

  const [filterIdx, setFilterIdx] = useState<number[] | null>(null);
  const [filteredIdxSet, setFilteredIdxSet] = useState<Set<number>>(
    () => new Set()
  );

  const getComments = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoadingComments(true);
    (async () => {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/get-posts`,
        {
          body: JSON.stringify({ username }),
          method: "POST",
          headers: {
            "content-type": "application/json",
          },
        }
      );
      const data = await res.json();
      setLoadingComments(false);
      if (res.status === 404) {
        setFormError(data.detail ?? "Something went wrong");
      } else {
        setSearchTerm("");
        setFilterIdx(null);
        setFilteredIdxSet(new Set());
        const commentsWithId = data.map((comment: any, i: number) => ({
          ...comment,
          id: i,
        }));
        setFormError(null);
        setComments(commentsWithId);
        setConfidence(null);
        setCurrentUser(username);
        search.current = elasticlunr(function () {
          this.addField("text");
          this.addField("subreddit");
        });
        for (const comment of commentsWithId) {
          search.current?.addDoc(comment);
        }
      }
    })();
  };

  const [predicting, setPredicting] = useState(false);

  const predict = () => {
    setPredicting(true);
    const commentsToInclude = comments.filter((_, i) => !filteredIdxSet.has(i));
    fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/predict`, {
      body: JSON.stringify({ posts: commentsToInclude }),
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((data) => {
        setConfidence(data);
        setPredicting(false);
      })
      .catch(() => {
        setConfidence(null);
        setPredicting(false);
      });
  };

  const update = useCallback((query: string) => {
    if (query === "") {
      setFilterIdx(null);
      return;
    }
    const indexes =
      search.current
        ?.search(query, {
          fields: {
            subreddit: { boost: 2 },
            text: { boost: 1 },
          },
          expand: true,
        })
        .map((result) => Number.parseInt(result.ref, 10)) ?? [];
    setFilterIdx(indexes);
  }, []);
  const debounced = useCallback(debounce(update, 500), []);

  const toggleVisibleComments = () => {
    if (filterIdx === null) {
      // toggle all visible on if there are more visible than hidden
      const turnOn =
        comments.length - filteredIdxSet.size < filteredIdxSet.size;
      const newSet = turnOn
        ? new Set<number>()
        : new Set<number>(Array.from({ length: comments.length }, (_, i) => i));
      setFilteredIdxSet(newSet);
    } else {
      const numSearchedCommentsDisabled = filterIdx.filter((i) =>
        filteredIdxSet.has(i)
      ).length;
      const totalSearched = filterIdx.length;
      const turnOn = numSearchedCommentsDisabled > totalSearched / 2;
      const newSet = new Set<number>(filteredIdxSet);
      if (turnOn) {
        filterIdx.forEach((i) => newSet.delete(i));
      } else {
        filterIdx.forEach((i) => newSet.add(i));
      }
      setFilteredIdxSet(newSet);
    }
  };

  const toggleDepressionSubreddits = () => {
    const newSet = new Set<number>(filteredIdxSet);
    const commentIdsInDepressionSubreddits = comments.map((comment, i) => {
      if (SUBREDDITS_RELATED_TO_DEPRESSION.has(comment.subreddit.toLowerCase())) {
        return i;
      }
      return null;
    }).filter((i) => i !== null) as number[];

    const numDepressionSubredditsDisabled = commentIdsInDepressionSubreddits.filter(
      (i) => newSet.has(i)
    ).length;

    const totalDepressionSubreddits = commentIdsInDepressionSubreddits.length;
    const turnOn = numDepressionSubredditsDisabled > totalDepressionSubreddits / 2;
    if (turnOn) {
      commentIdsInDepressionSubreddits.forEach((i) => newSet.delete(i));
    } else {
      commentIdsInDepressionSubreddits.forEach((i) => newSet.add(i));
    }

    setFilteredIdxSet(newSet);
  }

  useEffect(() => {
    debounced(searchTerm);
  }, [searchTerm]);

  return (
    <>
      <Head>
        <title>Depression Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        <Box w="100%" overflow="hidden" px="36" pt="20">
          <Heading fontSize="5xl">😔🤖 Depression Detection</Heading>
          <Text mt="1">
            This is a demo of a depression detection model trained on Reddit
            data.
          </Text>
          <FormControl
            isInvalid={typeof formError === "string"}
            mb="10"
            pl="4"
            borderLeft="3px solid"
            borderLeftColor="gray.200"
          >
            <form onSubmit={getComments}>
              <FormLabel display="block" mt="5">
                Username
              </FormLabel>
              <HStack w="40%">
                <Input
                  placeholder="MonLiH"
                  onChange={(e) => setUsername(e.target.value)}
                />
                <Button
                  px="8"
                  colorScheme={currentUser ? "gray" : "green"}
                  type="submit"
                  isLoading={loadingComments}
                  isDisabled={loadingComments}
                >
                  Get Comments
                </Button>
              </HStack>
              {formError ? (
                <FormErrorMessage color="red">{formError}</FormErrorMessage>
              ) : (
                <FormHelperText>
                  Enter the username to get comments for.
                </FormHelperText>
              )}
            </form>
          </FormControl>
          <Divider />
          {currentUser && (
            <HStack w="100%" h="100vh" pt="10" mt="-5">
              <VStack
                alignItems="baseline"
                h="100%"
                flexGrow={1.5}
                flexBasis={0}
                borderRight="1px solid"
                borderRightColor="gray.200"
                pr="5"
              >
                <HStack>
                  <Text fontSize="xl" fontWeight="bold" mr="2">
                    Comments of {currentUser ? `/u/${currentUser}` : "..."}
                  </Text>
                  <Tooltip
                    hasArrow
                    label="Toggle subreddits related to depression (to test early depression detection)."
                  >
                    <Button size="xs" colorScheme="green" onClick={toggleDepressionSubreddits}>Toggle Subreddits</Button>
                  </Tooltip>
                </HStack>
                <HStack w="100%">
                  <InputGroup>
                    <InputLeftElement pointerEvents={"none"}>
                      <Search />
                    </InputLeftElement>
                    <Input
                      type="text"
                      placeholder="Search"
                      onChange={(e) => {
                        setSearchTerm(e.target.value);
                      }}
                      value={searchTerm}
                    />
                  </InputGroup>
                  <Button onClick={toggleVisibleComments}>Toggle Hide</Button>
                </HStack>
                <Box flexGrow={1} maxWidth="50vw" overflow="scroll" pb="4">
                  <VStack alignItems="baseline">
                    {(
                      filterIdx?.map(
                        (index) => [comments[index], index] as [Comment, number]
                      ) ??
                      comments.map(
                        (comment, i) => [comment, i] as [Comment, number]
                      )
                    ).map(([comment, i]: [Comment, number], _) => (
                      <Box
                        key={i}
                        p="3"
                        border="1px solid"
                        borderColor="gray.200"
                        borderRadius="5"
                        position={"relative"}
                        w="100%"
                        color={filteredIdxSet.has(i) ? "gray.400" : "black"}
                        _hover={{
                          borderColor: "gray.500",
                          _after: {
                            // x emoji if not in filtered set otherwise check emoji
                            content: `""`,
                            bgImage: filteredIdxSet.has(i)
                              ? "url('/visible.svg')"
                              : "url('/hidden.svg')",
                            position: "absolute",
                            right: "10px",
                            top: "10px",
                            width: "24px",
                            height: "24px",
                          },
                        }}
                        onClick={() => {
                          const newSet = new Set(filteredIdxSet);
                          if (filteredIdxSet.has(i)) {
                            newSet.delete(i);
                          } else {
                            newSet.add(i);
                          }
                          setFilteredIdxSet(newSet);
                        }}
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
              <Box p="5" h="100%" flexGrow={1} flexBasis={0} w="100%">
                <Text fontSize="sm" mb="2">Predict using <b>{comments.length - filteredIdxSet.size}</b> comments:</Text>
                <Button
                  onClick={predict}
                  mb="4"
                  colorScheme={"green"}
                  disabled={predicting}
                  isLoading={predicting}
                >
                  Predict
                </Button>
                {confidence && (
                  // show bar graph
                  <Box w="100%">
                    <HStack>
                      <Text
                        fontWeight={
                          confidence[0] > confidence[1] ? "bold" : "normal"
                        }
                        w="120px"
                        flexGrow={0}
                      >
                        Not Depressed
                      </Text>
                      <Box
                        w={`${300 * confidence[0]}px`}
                        borderRightRadius="5"
                        h="10px"
                        bg="blue.500"
                      />
                      <Text>{round(confidence[0])}</Text>
                    </HStack>
                    <HStack>
                      <Text
                        fontWeight={
                          confidence[0] < confidence[1] ? "bold" : "normal"
                        }
                        w="120px"
                        flexGrow={0}
                      >
                        Depressed
                      </Text>
                      <Box
                        w={`${300 * confidence[1]}px`}
                        h="10px"
                        bg="red.500"
                        borderRightRadius="5"
                      />
                      <Text>{round(confidence[1])}</Text>
                    </HStack>
                  </Box>
                )}
              </Box>
            </HStack>
          )}
        </Box>
      </main>
    </>
  );
}
