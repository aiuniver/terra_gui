<template>
  <div class="toolbar project-training-toolbar">
    <div class="wrapper">
      <ul class="menu-section">
        <li data-type="charts" disabled="disabled">
          <span title="Графики" class="icon-training-charts"></span>
        </li>
        <li data-type="scatters" disabled="disabled">
          <span title="Скаттеры" class="icon-training-scatters"></span>
        </li>
        <li data-type="images" disabled="disabled">
          <span title="Изображения" class="icon-training-images"></span>
        </li>
        <li data-type="texts" disabled="disabled">
          <span title="Текст" class="icon-training-texts"></span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  name: "Toolbar"
}
</script>

<style scoped>

</style>