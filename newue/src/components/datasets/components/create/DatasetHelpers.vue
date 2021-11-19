<template>
  <div class="dataset-helpers">
    <div class="dataset-helpers-header">
      <p>Для создания датасета необходимо:</p>
    </div>
    <div class="dataset-helpers-list mt-5 ml-5">
      <ul class="dataset-helpers-list__menu">
        <li class="flex align-center">
          <span class="panel-text-gray">1. Выбрать входные чистые данные</span>
          <SvgContainer name="sloy-start-add" />
        </li>
        <li class="flex align-center">
          <span class="panel-text-gray">2. Задать обработчик</span>
          <SvgContainer name="sloy-middle-add" />
        </li>
        <li class="flex align-center">
          <span class="panel-text-gray">3. Соединить все в инпут вход один</span>
          <SvgContainer name="sloy-end-add" />
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
import SvgContainer from '@/components/app/SvgContainer.vue';
export default {
  components: { SvgContainer },
  name: 'DatasetHelpers',
};
</script>

<style lang="scss" scoped>
.dataset-helpers {
  &-list {
    &__menu {
      max-width: 320px;
      li {
        margin-bottom: 25px;
        span {
          margin-right: auto;
        }
      }
    }
  }
}
</style>