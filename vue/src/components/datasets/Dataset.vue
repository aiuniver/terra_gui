<template>
  <div class="board">
    <div class="wrapper">
      <Filters />
      <div class="project-datasets-block datasets" :style="height">
        <div class="title" @click="click('name')">Выберите датасет</div>
        <scrollbar >
          <div class="inner">
            <div class="dataset-card-container">
              <div class="dataset-card-wrapper">
                <template v-for="(dataset, key) of datasets">
                  <Card
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
import Card from "@/components/datasets/Card";

export default {
  components: {
    Card,
    Filters,
  },
  computed: {
    ...mapGetters({
      datasets: "datasets/getDatasets",
    }),
    height() {
      const filterHeight = this.$store.getters['settings/getFilterHeight']
      return this.$store.getters["settings/height"](filterHeight + 207);
    },
  },
  methods: {
    click(dataset){
      this.$store.dispatch('datasets/setSelect', dataset)
      // this.$store.dispatch('messages/setMessage', { message: `Выбран датасет «${dataset.name}»`})
    },
    change(value) {
      console.log(this.height)
      console.log(value)
      this.filterHeight = value
      console.log(this.height)
    }
  },
  mounted() {
    this.items = this.datasets.map((item) => {
      return item
    })
  },
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  width: 100%;
}
</style>