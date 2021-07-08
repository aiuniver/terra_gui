<template>
  <main class="page-datasets">
    <div class="container">
      <Dataset />
      <Params />
    </div>
  </main>
</template>

<script>
import Dataset from "@/components/datasets/Dataset.vue";
import Params from "@/components/datasets/params/Params.vue";

export default {
  name: "Datasets",
  components: {
    Dataset,
    Params,
  },
};
</script>
