<template>
  <div class="datasets">
    <p class="datasets-type flex align-center">
      <span class="mr-4">{{ list[selectedType] }}</span>
      <d-svg name="file-blank-outline" v-if="selectedType === 1" />
    </p>

    <DatasetsFilters
      :data="sortedDatasets"
      :selectedType="selectedType"
      :cardsDisplay="cardsDisplay"
      @changeFilter="handleChangeFilter"
      @changeDisplay="cardsDisplay = $event"
    />

    <scrollbar style="justify-self: stretch">
      <div v-if="cardsDisplay" class="datasets-cards">
        <DatasetCard @click.native="selectDataset(item)" v-for="(item, idx) in sortedList" :key="idx" :dataset="item" />
      </div>
      <Table v-else :data="sortedDatasets" :selectedType="selectedType" />
    </scrollbar>
  </div>
</template>

<script>
import DatasetCard from '@/components/datasets/choice/DatasetCard';
import Table from '@/components/datasets/choice/Table';
import DatasetsFilters from '@/components/datasets/choice/DatasetsFilters';

export default {
  name: 'Datasets',
  props: ['datasets', 'selectedType'],
  components: {
    DatasetCard,
    Table,
    DatasetsFilters,
  },
  data: () => ({
    list: ['Недавние датасеты', 'Проектные датасеты', 'Датасеты Terra'],
    cardsDisplay: true,
    selectedSort: {},
  }),
  methods: {
    selectDataset(item) {
      this.$store.dispatch('datasets/selectDataset', item);
    },
    handleChangeFilter(sort) {
      this.selectedSort = sort;
    },
    randomDate(start, end, startHour, endHour) {
      let date = new Date(+start + Math.random() * (end - start));
      let hour = (startHour + Math.random() * (endHour - startHour)) | 0;
      date.setHours(hour);
      return date;
    },
  },
  computed: {
    sortedDatasets() {
      return this.datasets.map(el => {
        return {
          ...el,
          size: el.size ? el.size : 'Предустановленный',
          date: el.date ? el.date : this.randomDate(10, 30, 10, 15),
        };
      });
    },

    sortedList() {
      const sortDatasets = this.datasets;
      if (this.selectedSort.value === 'alphabet' || Object.keys(this.selectedSort).length === 0)
        return sortDatasets.sort((a, b) => a.name.localeCompare(b.name));
      if (this.selectedSort.value === 'alphabet_reverse') return sortDatasets.sort((a, b) => b.name.localeCompare(a.name));
      return sortDatasets;
    },
  },
};
</script>

<style lang="scss" scoped>
.datasets {
  padding: 30px 30px;
  padding: 30px 30px 0 30px;
  width: 100%;
  height: 100%;
  max-height: 865px;
  display: flex;
  flex-direction: column;
  gap: 30px;
  &-cards {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }
  &-type {
    font-size: 14px;
    font-weight: 600;
    height: 20px;
  }
}
</style>