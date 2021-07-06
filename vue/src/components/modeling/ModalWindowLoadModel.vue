<template>
  <div id="modal-window">
    <div class="overlay"></div>
    <div class="inner">
      <div class="header">
        <div class="title"></div>
        <div class="close" title="[ESC]"></div>
      </div>
      <div class="container">
        <div class="wrapper">
          <div id="modal-window-load-model" class="modal-window-container">
            <div class="models-data">
              <div class="models-list">
                <div class="custom-scrollbar-wrapper">
                  <ul class="loaded-list">
                    <li
                        v-for="item in loaded_list"
                        :key="item.name"
                        :data-name="item.name"
                        :data-is_terra="item.is_terra"
                    >
                      <span>{{ item.name }}</span>
                    </li>
                  </ul>
                </div>
              </div>
              <div class="model-arch">
                <div class="wrapper hidden">
                  <div class="modal-arch-info">
                    <div class="model-arch-info-param name">Name: <span></span></div>
                    <div class="model-arch-info-param input_shape">Input shape: <span></span></div>
                    <div class="model-arch-info-param datatype">Datatype: <span></span></div>
                  </div>
                  <div class="model-arch-img"><img src="" alt="" /></div>
                  <div class="model-save-arch-btn"><button>Загрузить</button></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "ModalWindowLoadModel",
  data: () => ({
    loaded_list: [
      {is_terra: true, name: "mnist"},
      {is_terra: true, name: "cifar10"},
      {is_terra: true, name: "cifar100"},
      {is_terra: true, name: "PSP"},
      {is_terra: false, name: "Custom mnist"},
      {is_terra: false, name: "Custom GG"}
    ]
  })
}
</script>

<style scoped>

</style>