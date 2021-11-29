<template>
  <main class="page-datasets">
    <div class="cont">
      <div class="datasets-menu">
        <div class="datasets-menu-list">
          <div
            @click="selectedType = 0"
            :class="['datasets-menu-list__item', { 'datasets-menu-list__item--selected': selectedType === 0 }]"
          >
            <SvgContainer name="clock" />
            <span>Недавние</span>
          </div>
          <div
            @click="selectedType = 1"
            :class="['datasets-menu-list__item', { 'datasets-menu-list__item--selected': selectedType === 1 }]"
          >
            <SvgContainer name="file-outline" />
            <span>Проектные</span>
          </div>
          <div
            @click="selectedType = 2"
            :class="['datasets-menu-list__item', { 'datasets-menu-list__item--selected': selectedType === 2 }]"
          >
            <SvgContainer name="world" />
            <span>Terra</span>
          </div>
        </div>
        <hr />
        <div class="datasets-menu-categories">
          <ul class="datasets-menu-categories__item">
            <li>Изображения</li>
            <li>Машины</li>
            <li>Круглые</li>
            <li>Самолеты</li>
            <li>Квадраты</li>
          </ul>
          <ul class="datasets-menu-categories__item">
            <li>Видео</li>
            <li>Машины</li>
            <li>Круглые</li>
            <li>Самолеты</li>
            <li>Квадраты</li>
          </ul>
          <ul class="datasets-menu-categories__item">
            <li>Текст</li>
            <li>Машины</li>
            <li>Круглые</li>
            <li>Самолеты</li>
            <li>Квадраты</li>
          </ul>
        </div>
      </div>
      <Datasets :datasets="datasets" :selectedType="selectedType" />
      <BasePanel v-if="Object.keys(choiceDataset).length"> 
        <BasePanelContent>
          <template #header>Общее</template>
           <template #content>
            <ul>
              <li>
                <span>Название:</span>
                <span>{{ choiceDataset.name }} </span>
              </li>
              <li>
                <span>Вес:</span>
              </li>
              <li>
                <span>Создатель:</span>
              </li>
            </ul>
          </template>
        </BasePanelContent>
        <BasePanelContent>
          <template #header>Отношение</template>
           <template #content>
            asd
          </template>
        </BasePanelContent>
        <BasePanelContent>
          <template #header>Размерности типов данных </template>
           <template #content>
            asd
          </template>
        </BasePanelContent>
         <BasePanelContent>
          <template #header>Пример файлов</template>
           <template #content>
            asd
          </template>
        </BasePanelContent>
      </BasePanel>
    </div>
  </main>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  name: 'Choice',
  components: {
    Datasets: () => import('@/components/datasets/Datasets.vue'),
  },
  data: () => ({
    selectedType: 2,
  }),
  computed: {
    ...mapGetters({
      datasets: 'datasets/getDatasets',
      choiceDataset: 'datasets/choiceDataset'
    }),
  },
};
</script>

<style lang="scss" scoped>
.cont {
  // background-color: $color-dark;
  padding: 0;
  display: flex;
  height: 100%;
}

.datasets-menu {
  width: 175px;
  border-right: 1px solid $color-black;
  padding-top: 18px;
  hr {
    border: none;
    border-bottom: 1px solid $color-dark-gray;
    margin: 20px;
  }
  &-list {
    &__item {
      display: flex;
      align-items: center;

      padding: 10px 0 10px 20px;
      color: $color-gray-blue;
      font-size: 14px;
      cursor: pointer;

      span{
        margin-left: 15px;
      }

      &--selected {
        background-color: $color-black;
      }

      &:hover {
        color: #65b9f4;
        .ci-world {
          background-color: #65b9f4;
        }
      }
      &.selected {
        background-color: var(--color-light-gray);
      }
      input {
        display: none;
      }
    }
  }
  &-categories {
    font-size: 14px;
    padding-left: 20px;
    &__item {
      margin-bottom: 10px;
      li {
        &:first-child {
          padding-bottom: 5px;
          color: $color-white;
          cursor: default;
        }
        color: $color-gray;
        padding-top: 5px;
        cursor: pointer;
      }
    }
  }
}
</style>