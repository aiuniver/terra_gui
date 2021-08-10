<template>
  <div class="board">
    <div class="wrapper">
      <Filters />
      <div class="project-datasets-block datasets" :style="height">
        <div class="title" @click="click('name')">Выберите датасет</div>
        <scrollbar  >
          <div class="inner">
            <div class="dataset-card-container">
              <div class="dataset-card-wrapper">
                <template v-for="(dataset, key) of datasets">
                  <CardDataset
                    :dataset="dataset"
                    :key="key"
                    @clickCard="click"
                  />
                </template>
              </div>
            </div>
          </div>
        </scrollbar>
      </div>
    </div>
  </div>
</template>

<script>
import Filters from "@/components/datasets/Filters.vue";
import { mapGetters } from "vuex";
import CardDataset from "@/components/datasets/cards/CardDataset";

export default {
  components: {
    CardDataset,
    Filters,
  },
  data: () => ({
    hight: 0
  }),
  computed: {
    ...mapGetters({
      datasets: "datasets/getDatasets",
    }),
    height() {
      return this.$store.getters['settings/height']({ deduct: 'filter', padding: 52, clean: true })
    },
  },
  methods: {
    click(dataset){
      this.$store.dispatch('datasets/setSelect', dataset)
      // this.$store.dispatch('messages/setMessage', { message: `Выбран датасет «${dataset.name}»`})
    },
  },
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  width: 100%;
  background-color: #17212b;
}
</style>