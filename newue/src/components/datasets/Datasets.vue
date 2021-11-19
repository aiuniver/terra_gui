<template>
  <div class="datasets">
    <p class="datasets-type flex align-center">
      <span class="mr-4">{{ list[selectedType] }}</span>
      <SvgContainer name="file-blank-outline" v-if="selectedType === 1" />
    </p>

    <DatasetsFilters
      :data="sortedDatasets"
      :selectedType="selectedType"
      :cardsDisplay="cardsDisplay"
      @changeFilter="handleChangeFilter"
      @changeDisplay="cardsDisplay = $event"
    />

    <scrollbar style="justify-self: stretch; height: 695px;">
      <div v-if="cardsDisplay" class="datasets-cards">
        <DatasetCard v-for="(item, idx) in sortedList" :key="idx" :dataset="item" />
      </div>
      <Table v-else :data="sortedDatasets" :selectedType="selectedType" />
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: 'Datasets',
  props: ['datasets', 'selectedType'],
  components: {
    DatasetCard: () => import('@/components/datasets/components/choice/DatasetCard.vue'),
    SvgContainer: () => import('@/components/app/SvgContainer.vue'),
    Table: () => import('@/components/datasets/components/choice/Table.vue'),
    DatasetsFilters: () => import('@/components/datasets/components/choice/DatasetsFilters.vue'),
  },
  data: () => ({
    list: ['Недавние датасеты', 'Проектные датасеты', 'Датасеты Terra'],
    cardsDisplay: true,
    selectedSort: {}
  }),
  methods:{
    handleChangeFilter(sort){
      this.selectedSort = sort
    },  
    randomDate(start, end, startHour, endHour) {
      let date = new Date(+start + Math.random() * (end - start));
      let hour = startHour + Math.random() * (endHour - startHour) | 0;
      date.setHours(hour);
      return date;
    }
  },
  computed: {
    sortedDatasets() {
      return this.datasets.map(el => {
        return {
          ...el,
          size: el.size ? el.size : 'Предустановленный',
          date: el.date ? el.date : this.randomDate(10, 30, 10, 15),
        }
      });
    },

    sortedList() {
      const sortDatasets = this.datasets;
      if (this.selectedSort.value === 'alphabet' || Object.keys(this.selectedSort).length === 0) 
        return sortDatasets.sort((a, b) => a.name.localeCompare(b.name));
      if (this.selectedSort.value === 'alphabet_reverse')
        return sortDatasets.sort((a, b) => b.name.localeCompare(a.name));
      return sortDatasets
    },

  },
};
</script>

<style lang="scss" scoped>
.datasets {
  padding: 30px 0 0 40px;
  width: 100%;
  height: 100%;
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