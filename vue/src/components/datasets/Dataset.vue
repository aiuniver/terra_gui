<template>
  <div class="board">
    <div class="wrapper">
      <Filters />
      <div class="project-datasets-block datasets" :style="height">
        <div class="title mb-4" @click="click('name')">Выберите датасет</div>
        <scrollbar>
          <div class="inner">
            <div class="dataset-card-container">
              <div class="dataset-card-wrapper">
                <template v-for="(dataset, key) of datasets">
                  <CardDataset
                    :dataset="dataset"
                    :key="key"
                    :cardIndex="key"
                    :loaded="isLoaded(dataset)"
                    @click="click"
                    @remove="remove"
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
import Filters from '@/components/datasets/Filters.vue';
import { mapGetters } from 'vuex';
import CardDataset from '@/components/datasets/cards/CardDataset';

export default {
  components: {
    CardDataset,
    Filters,
  },
  data: () => ({
    hight: 0,
  }),
  computed: {
    ...mapGetters({
      datasets: 'datasets/getDatasets',
      project: 'projects/getProject',
    }),
    height() {
      return this.$store.getters['settings/height']({ deduct: 'filter', padding: 52, clean: true });
    },
  },
  mounted() {
    document.addEventListener('click', () => {
      this.$store.dispatch('datasets/setSelect', 0);
      this.$store.dispatch('datasets/setSelectedIndex', null);
    });
  },
  methods: {
    isLoaded(dataset) {
      // console.log(this.project?.dataset?.alias)
      return this.project?.dataset?.alias === dataset.alias;
    },
    click(dataset, key) {
      if (!dataset.training_available) return;
      if (!this.isLoaded(dataset)) {
        // console.log(dataset, key);
        this.$store.dispatch('datasets/setSelect', dataset);
        this.$store.dispatch('datasets/setSelectedIndex', key);
        return;
      }
      this.$store.dispatch('datasets/setSelect', 0);
      this.$store.dispatch('datasets/setSelectedIndex', null);
      // let card = e.path.filter(element => element.className == "dataset-card")[0]
      // this.$store.dispatch('messages/setMessage', { message: `Выбран датасет «${dataset.name}»`})
    },
    async remove({ name, alias, group }) {
      try {
        await this.$Modal.confirm({
          title: 'Внимание!',
          content: `Вы действительно желаете удалить датасет "${name}" ?`,
          width: 300,
        });
        this.$store.dispatch('settings/setOverlay', true);
        await this.$store.dispatch('datasets/deleteDataset', { alias, group });
        this.$store.dispatch('settings/setOverlay', false);
      } catch (error) {
        this.$store.dispatch('settings/setOverlay', false);
        console.log(error);
      }
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
