<template>
  <main class="page-datasets">
    <div class="container">
      <div class="row">
        <div :class="full ? 'col-3' : 'col-18'">
          <Dataset />
        </div>
        <div :class="full ? 'col-21' : 'col-6'">
          <ParamsFull v-if="full" />
          <Params v-else />
        </div>
      </div>
    </div>
  </main>
</template>

<script>
import { mapGetters } from 'vuex'
import Dataset from "@/components/datasets/Dataset.vue";
import Params from "@/components/datasets/params/Params.vue";
import ParamsFull from "@/components/datasets/params/ParamsFull.vue";

export default {
  name: "Datasets",
  components: {
    Dataset,
    Params,
    ParamsFull,
  },
  data: () => ({
  }),
  computed: {
    ...mapGetters({
      full: 'datasets/getFull'
    })
  }
};
</script>
