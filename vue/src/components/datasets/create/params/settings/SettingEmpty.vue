<template>
  <div class="panel-settings-empty">
    <div class="panel-settings-empty__title mb-4">Для создания датасета необходимо:</div>
    <div class="panel-settings-empty__info pl-4">
      <ul>
        <li>
          <span>1. Выбрать входные чистые данные</span>
          <d-svg name="sloy-start-add" />
        </li>
        <li>
          <span>2. Задать обработчик</span>
          <d-svg name="sloy-middle-add" />
        </li>
        <li>
          <span>3. Соединить все в инпут вход один</span>
          <d-svg name="sloy-handler-add" />
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {};
</script>

<style lang="scss">
.panel-settings-empty {
  &__info {
    li {
      font-size: 14px;
      line-height: 140%;
      color: #a7bed3;
      padding: 10px 0;
      display: flex;
      align-items: center;
      span {
        margin-right: 20px;
      }
    }
  }
}
</style>