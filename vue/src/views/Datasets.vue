<template>
  <main class="page-datasets">
    <div class="container-fluid">
      <div class="main">
        <div class="one">
          <Dataset />
        </div>
        <div class="two">
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

<style lang="scss" scoped>
.container-fluid{
  background: #17212B;
  padding: 0;
}

.main {
  display: flex
}
.one {
  flex-shrink: 1;
}
.two {
  flex-shrink: 0;
  width: 400px;
}
</style>