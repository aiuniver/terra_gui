<template>
  <main class="page-datasets">
    <div class="container">
      <div class="row">
        <div :class="full ? 'col-18' : 'col-3'">
          <Dataset />
        </div>
        <div :class="full ? 'col-6' : 'col-21'">
          <Params v-if="full" />
          <ParamsFull v-else />
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
